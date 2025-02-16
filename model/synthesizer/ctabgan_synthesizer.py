import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.optim as optim
from torch.optim import Adam
from torch.nn import functional as F
from torch.nn import (Dropout, LeakyReLU, Linear, Module, ReLU, Sequential,
Conv2d, ConvTranspose2d, Sigmoid, init, BCELoss, CrossEntropyLoss,SmoothL1Loss,LayerNorm)
from model.synthesizer.transformer import ImageTransformer,DataTransformer

from model.privacy_utils.rdp_accountant import compute_rdp, get_privacy_spent
from tqdm import tqdm


def apply_activate(data, output_info):
    data_t = []
    st = 0
 
    for activations in output_info:
        for activation in activations:
            output_size, activation_type, _= activation
            ed = st + output_size
            if activation_type == 'tanh':
                data_t.append(torch.tanh(data[:, st:ed]))
            elif activation_type == 'softmax':
                data_t.append(F.gumbel_softmax(data[:, st:ed], tau=0.2))
            else:
                raise ValueError('Activation function not supported')
            st = ed
    return torch.cat(data_t, dim=1)



def random_choice_prob_index_sampling(probs,col_idx):
    option_list = []
    for i in col_idx:
        pp = probs[i]
        option_list.append(np.random.choice(np.arange(len(probs[i])), p=pp))
    
    #TODO: rewrite it in vector form (this only used in the sampling)
    return np.array(option_list).reshape(col_idx.shape)

def random_choice_prob_index(a, axis=1):
    r = np.expand_dims(np.random.rand(a.shape[1 - axis]), axis=axis)
    return (a.cumsum(axis=axis) > r).argmax(axis=axis)

def maximum_output_interval(output_info):
    max_interval = 0
    for activations in output_info:
        for activation in activations:
            output_size = activation[0]
        max_interval = max(max_interval, output_size)
    return max_interval

class Cond(object):
    def __init__(self, data, output_info):
       
        self.model = []
        st = 0
        counter = 0
        for activations in output_info:
            for activation in activations:
                output_size, activation_type, _= activation
                if activation_type not in ['tanh', 'softmax']: 
                    raise ValueError('Activation function not supported')
                
                if activation_type == 'softmax':
                    ed = st + output_size
                    counter += 1
                    self.model.append(np.argmax(data[:, st:ed], axis=-1))

                st += output_size
                
        self.interval = []
        self.n_col = 0  
        self.n_opt = 0  
        st = 0
        self.p = np.zeros((counter , maximum_output_interval(output_info)))  
        
        self.col_to_cat_index = []  #for converting column index to category index when conditioning on generated data
        
        for col_index, activations in enumerate(output_info):
            categorical_index = None
            for activation in activations:
                output_size, activation_type, _= activation
                if activation_type not in ['tanh', 'softmax']: 
                    raise ValueError('Activation function not supported')
                
                if activation_type == 'softmax': # 
                    ed = st + output_size
                    tmp = np.sum(data[:, st:ed], axis=0)
                    tmp = np.log(tmp + 1)
                    tmp = tmp / np.sum(tmp)
                    self.p[self.n_col, :output_size] = tmp
                    self.interval.append((self.n_opt, output_size))
                    self.n_opt += output_size
                    categorical_index = self.n_col
                    self.n_col += 1
                    
                
                st += output_size

            self.col_to_cat_index.append(categorical_index)
                
                
        self.interval = np.asarray(self.interval)
        
    def sample_train(self, batch):
        if self.n_col == 0: return None
    
        idx = np.random.choice(np.arange(self.n_col), batch)
        vec = np.zeros((batch, self.n_opt), dtype='float32')
        
        mask = np.zeros((batch, self.n_col), dtype='float32')
        mask[np.arange(batch), idx] = 1  
        opt1prime = random_choice_prob_index(self.p[idx]) 
        for i in np.arange(batch):
            vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1

        vec[np.arange(batch), self.interval[idx, 0] + opt1prime] = 1 # The zero is placed at the the start interval of the idx + the offset for the categorical value

            
        return vec, mask, idx, opt1prime

    def sample(self, batch,column_index,column_value_index):
        if self.n_col == 0: return None #TODO: does this mean we need at least one categorical column?

        if column_index is not None and column_value_index is not None: #If condition is specified we generate sample with such a condition
            return self.sample_condition(batch, column_index, column_value_index)

        idx = np.random.choice(np.arange(self.n_col), batch)
        vec = np.zeros((batch, self.n_opt), dtype='float32')
        opt1prime = random_choice_prob_index_sampling(self.p,idx)
        vec[np.arange(batch), self.interval[idx, 0] + opt1prime] = 1
            
        return vec

    def sample_condition(self, batch, column_index, column_value_index):
        
        cat_index = self.col_to_cat_index[column_index]
        idx = np.full(batch, cat_index) 
        vec = np.zeros((batch, self.n_opt), dtype='float32')
        vec[np.arange(batch), self.interval[idx, 0] + column_value_index] = 1

        return vec # TODO: make it such you do not have to specify

    

def cond_loss(data, output_info, c, m):
    loss = []
    st = 0
    st_c = 0

    for activations in output_info:
        for activation in activations:
            output_size, activation_type, _= activation
            if activation_type not in ['tanh', 'softmax']: 
                raise ValueError('Activation function not supported')
                
            if activation_type == 'softmax':
                ed = st + output_size
                ed_c = st_c + output_size
                tmp = F.cross_entropy(
                data[:, st:ed],
                torch.argmax(c[:, st_c:ed_c], dim=1),
                reduction='none')
                loss.append(tmp)
                st_c = ed_c
            
            st += output_size

    loss = torch.stack(loss, dim=1)
    return (loss * m).sum() / data.size()[0]

def compute_constraint_penalty(data):
    

    # Compute the difference between the first and second columns
    diff = data[:, 0] - data[:, 1]

    # Apply ReLU to the negative differences to get the penalty
    penalty = F.relu(-diff)

    # Sum the penalties to get the total constraint penalty
    total_penalty = penalty.sum()

    return total_penalty


 
    

class Sampler(object):
    def __init__(self, data, output_info):
        super(Sampler, self).__init__()
        self.data = data
        self.model = []
        self.n = len(data)
        st = 0
        
        for activations in output_info:
            for activation in activations:
                output_size, activation_type, _= activation
                if activation_type == 'tanh':
                    st += output_size
                    continue
                elif activation_type == 'softmax':
                    ed = st + output_size
                    tmp = []
                    for j in range(output_size):
                        tmp.append(np.nonzero(data[:, st + j])[0])
                    self.model.append(tmp)
                    st = ed
                else:
                    raise ValueError('Activation function not supported')
                
    def sample(self, n, col, opt):
        if col is None:
            idx = np.random.choice(np.arange(self.n), n)
            return self.data[idx]
        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self.model[c][o]))
        return self.data[idx]

class Discriminator(Module):
    def __init__(self, side, layers):
        super(Discriminator, self).__init__()
        self.side = side
        info = len(layers)-2
        self.seq = Sequential(*layers)
        self.seq_info = Sequential(*layers[:info])

    def forward(self, input):
        return (self.seq(input)), self.seq_info(input)

class Generator(Module):
    def __init__(self, side, layers):
        super(Generator, self).__init__()
        self.side = side
        self.seq = Sequential(*layers)

    def forward(self, input_):
        return self.seq(input_)

def determine_layers_disc(side, num_channels):
    assert side >= 4 and side <= 64

    layer_dims = [(1, side), (num_channels, side // 2)]

    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

    layerNorms = []
    num_c = num_channels
    num_s = side / 2
    for l in range(len(layer_dims) - 1):
        layerNorms.append([int(num_c), int(num_s), int(num_s)])
        num_c = num_c * 2
        num_s = num_s / 2

    layers_D = []

    for prev, curr, ln in zip(layer_dims, layer_dims[1:], layerNorms):
        layers_D += [
            Conv2d(prev[0], curr[0], 4, 2, 1, bias=False),
            LayerNorm(ln),
            LeakyReLU(0.2, inplace=True),
        ]

    layers_D += [Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0), ReLU(True)] 

    return layers_D

def determine_layers_gen(side, random_dim, num_channels):
    assert side >= 4 and side <= 64

    layer_dims = [(1, side), (num_channels, side // 2)]

    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

    layerNorms = []

    num_c = num_channels * (2 ** (len(layer_dims) - 2))
    num_s = int(side / (2 ** (len(layer_dims) - 1)))
    for l in range(len(layer_dims) - 1):
        layerNorms.append([int(num_c), int(num_s), int(num_s)])
        num_c = num_c / 2
        num_s = num_s * 2

    layers_G = [ConvTranspose2d(random_dim, layer_dims[-1][0], layer_dims[-1][1], 1, 0, output_padding=0, bias=False)]

    for prev, curr, ln in zip(reversed(layer_dims), reversed(layer_dims[:-1]), layerNorms):
        layers_G += [LayerNorm(ln), ReLU(True), ConvTranspose2d(prev[0], curr[0], 4, 2, 1, output_padding=0, bias=True)]
    return layers_G

def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1)).view(val.size(0), 1)
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so)*low + (torch.sin(val*omega)/so) * high
    
    return res

def calc_gradient_penalty_slerp(netD, real_data, fake_data, transformer, device='cpu', lambda_=10):
    batchsize = real_data.shape[0]
    alpha = torch.rand(batchsize, 1,  device=device)
    interpolates = slerp(alpha, real_data, fake_data)
    interpolates = interpolates.to(device)
    interpolates = transformer.transform(interpolates)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates,_ = netD(interpolates) 

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0] 
    
    gradients_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradients_norm - 1) ** 2).mean() * lambda_
    
    return gradient_penalty

def weights_init(m):
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0)

class CTABGANSynthesizer:
    def __init__(self,
                 class_dim=(256, 256, 256, 256),
                 random_dim=100,
                 num_channels=64,
                 l2scale=1e-5,
                 batch_size=500):
                 

        self.random_dim = random_dim
        self.class_dim = class_dim
        self.num_channels = num_channels
        self.dside = None
        self.gside = None
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, prepared_data, data_prep, categorical, mixed, general,epochs=100):

        
        self.transformer = DataTransformer(prepared_data, data_prep, categorical, mixed, general)
       
        train_data = self.transformer.transform(prepared_data)
        data_sampler = Sampler(train_data, self.transformer.get_output_info())
        data_dim = self.transformer.get_output_dim()
        self.cond_generator = Cond(train_data, self.transformer.get_output_info())
        		
        sides = [4, 8, 16, 24, 32, 64] #TODO: this coulc be made faster and computaionally faster (use math instead of this)
        col_size_d = data_dim + self.cond_generator.n_opt #TODO: what happens here why datafim + n_opt
        for i in sides:
            if i * i >= col_size_d:
                self.dside = i
                break
        
        sides = [4, 8, 16, 24, 32, 64]
        col_size_g = data_dim
        for i in sides:
            if i * i >= col_size_g:
                self.gside = i
                break
		

        layers_G = determine_layers_gen(self.gside, self.random_dim+self.cond_generator.n_opt, self.num_channels)
        layers_D = determine_layers_disc(self.dside, self.num_channels)
        
        self.generator = Generator(self.gside, layers_G).to(self.device)
        discriminator = Discriminator(self.dside, layers_D).to(self.device)
        optimizer_params = dict(lr=2e-4, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.l2scale)
        optimizerG = Adam(self.generator.parameters(), **optimizer_params)
        optimizerD = Adam(discriminator.parameters(), **optimizer_params)

        st_ed = None
        classifier=None
        optimizerC= None

        target_index = None #TODO: remove this when finding solution to this
        
        self.generator.apply(weights_init)
        discriminator.apply(weights_init)

        self.Gtransformer = ImageTransformer(self.gside)       
        self.Dtransformer = ImageTransformer(self.dside)
        
        epsilon = 0
        epoch = 0
        steps = 0
        ci = 5
        
        steps_per_epoch = max(1, len(train_data) // self.batch_size)
        for i in tqdm(range(epochs)):
            for id_ in range(steps_per_epoch):
				
                
                for _ in range(ci):
                    noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
                    condvec = self.cond_generator.sample_train(self.batch_size)

                    c, m, col, opt = condvec
                    c = torch.from_numpy(c).to(self.device)
                    m = torch.from_numpy(m).to(self.device)
                    noisez = torch.cat([noisez, c], dim=1)
                    noisez =  noisez.view(self.batch_size,self.random_dim+self.cond_generator.n_opt,1,1)
                    
                    perm = np.arange(self.batch_size)
                    np.random.shuffle(perm)
                    real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
                    c_perm = c[perm]
                    
                    real = torch.from_numpy(real.astype('float32')).to(self.device)
                    
                    fake = self.generator(noisez)
                    faket = self.Gtransformer.inverse_transform(fake)
                    fakeact = apply_activate(faket, self.transformer.get_output_info())
                    
                    fake_cat = torch.cat([fakeact, c], dim=1)
                    real_cat = torch.cat([real, c_perm], dim=1)
                    
                    real_cat_d = self.Dtransformer.transform(real_cat)
                    fake_cat_d = self.Dtransformer.transform(fake_cat)
                    
                    optimizerD.zero_grad()
                    
                    d_real,_ = discriminator(real_cat_d)
                    

                    d_real = -torch.mean(d_real)
                    d_real.backward() 
                    

                    d_fake,_ = discriminator(fake_cat_d)
                    
                    d_fake = torch.mean(d_fake)

                    d_fake.backward() 
                    
                    pen = calc_gradient_penalty_slerp(discriminator, real_cat, fake_cat,  self.Dtransformer , self.device)

                    pen.backward()
                
                    optimizerD.step()
                    
                noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
                
                condvec = self.cond_generator.sample_train(self.batch_size)

                c, m, col, opt = condvec
                c = torch.from_numpy(c).to(self.device)
                m = torch.from_numpy(m).to(self.device)
                noisez = torch.cat([noisez, c], dim=1)
                noisez =  noisez.view(self.batch_size,self.random_dim+self.cond_generator.n_opt,1,1)

                optimizerG.zero_grad()

                fake = self.generator(noisez)
                faket = self.Gtransformer.inverse_transform(fake)
                fakeact = apply_activate(faket, self.transformer.get_output_info())

                fake_cat = torch.cat([fakeact, c], dim=1) 
                fake_cat = self.Dtransformer.transform(fake_cat)
                    
                y_fake,info_fake = discriminator(fake_cat)
                
                cross_entropy = cond_loss(faket, self.transformer.get_output_info(), c, m)
                # ata, meta, components_list, ordering, model, n_clusters, devic
                #inverse2, _ = self.transformer.inverse_transform_static(fakeact,self.transformer,self.device)
                #constraint_penalty = compute_constraint_penalty(inverse2)
                #t4 = self.transformer.inverse_transform(fakeact.detach().cpu())
                
                constraint_constant = 1e-1
                """
                fake_cat2 = fakeact
                fake_cat2 = fake_cat2.detach().cpu().numpy()
                fake_cat2_norm = self.transformer.inverse_transform(fake_cat2)
                """
                _,info_real = discriminator(real_cat_d)

                #print("Cross Entropy: ", cross_entropy, "\nConstraint Penalty: ", constraint_penalty, "\nThe...", -torch.mean(y_fake))
        

                g = -torch.mean(y_fake) + cross_entropy #+ constraint_constant * constraint_penalty
                g.backward(retain_graph=True)
                loss_mean = torch.norm(torch.mean(info_fake.view(self.batch_size,-1), dim=0) - torch.mean(info_real.view(self.batch_size,-1), dim=0), 1)
                loss_std = torch.norm(torch.std(info_fake.view(self.batch_size,-1), dim=0) - torch.std(info_real.view(self.batch_size,-1), dim=0), 1)
                loss_info = loss_mean + loss_std 
                loss_info.backward()
                optimizerG.step()

                                
            epoch += 1

            
   



    def sample(self, n, column_index = None,column_value_index = None): #TODO: move the column value and indexes to other places
        
        self.generator.eval()

        output_info = self.transformer.get_output_info()
        steps = n // self.batch_size + 1
        
        data = []
        
        for i in range(steps):
            noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
            condvec = self.cond_generator.sample(self.batch_size, column_index,column_value_index)
            c = condvec
            c = torch.from_numpy(c).to(self.device)
            noisez = torch.cat([noisez, c], dim=1)
            noisez =  noisez.view(self.batch_size,self.random_dim+self.cond_generator.n_opt,1,1)
                
            fake = self.generator(noisez)
            faket = self.Gtransformer.inverse_transform(fake)
            fakeact = apply_activate(faket,output_info)
            data.append(fakeact.detach().cpu().numpy())


        data = np.concatenate(data, axis=0)
        result,resample = self.transformer.inverse_transform(data)
        
        while len(result) < n:
            data_resample = []    
            steps_left = resample// self.batch_size + 1
            
            for i in range(steps_left):
                noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
                condvec = self.cond_generator.sample(self.batch_size, column_index,column_value_index)
                c = condvec
                c = torch.from_numpy(c).to(self.device)
                noisez = torch.cat([noisez, c], dim=1)
                noisez =  noisez.view(self.batch_size,self.random_dim+self.cond_generator.n_opt,1,1)
                    
                fake = self.generator(noisez)
                faket = self.Gtransformer.inverse_transform(fake)
                fakeact = apply_activate(faket, output_info)
                data_resample.append(fakeact.detach().cpu().numpy())

            data_resample = np.concatenate(data_resample, axis=0)

            res,resample = self.transformer.inverse_transform(data_resample)
            result  = np.concatenate([result,res],axis=0)
        
        return result[0:n]

