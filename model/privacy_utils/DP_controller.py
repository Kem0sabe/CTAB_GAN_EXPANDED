
from .rdp_accountant import compute_rdp, get_privacy_spent,compute_rdp_single,find_sigma_for_target_epsilon

class DP_controller:
    def __init__(self,data_size, batch_size, total_steps, epsilon_budget,delta=None,clip_coeff = 1, sigma = None,device = 'cpu',verbose = True):
        self.data_size = data_size
        self.batch_size = batch_size
        self.total_steps = total_steps # might step the discriminator more times in one epoch

        self.epsilon_budget = epsilon_budget
        self.delta = delta
        self.clip_coeff = clip_coeff
        self.sigma = sigma
        self.device = device
        self.verbose = verbose


        if not isinstance(batch_size, int) or batch_size < 1: 
            raise ValueError("Batch size must be a positive integer")

        if not isinstance(total_steps, int) or total_steps < 1:
            raise ValueError("Total steps must be a positive integer")
        
        self.q = batch_size/data_size if batch_size < data_size else 1 # If the batch size is set greater and the toal data size, then the whole data is used


        max_lmbd = 4095
        self.lmbds = range(2, max_lmbd + 1)

        if clip_coeff is None: raise ValueError("Clip coefficient cannot be set to None")

        self.delta = self._get_optimal_delta(self.data_size)

        self.sigma = self._get_optimal_sigma(self.q, self.total_steps,self.epsilon_budget, self.delta, self.sigma)

        self.rdp = compute_rdp_single(self.q, self.sigma, self.lmbds)

    

    def _get_optimal_delta(self,data_size):
        if self.delta:
            if self.verbose:
                print(f"Using provided delta: {self.delta}")
            return self.delta
        else:
            if self.verbose:
                print(f"No delta provided, setting delta: {0.1/data_size}")
            return 0.1/data_size


    def _get_optimal_sigma(self,q,total_steps, epsilon_budget, delta, sigma):
        if delta is None: ValueError("Delta must be provided")
        if sigma:
            if self.verbose:
                print(f"Using provided sigma: {sigma}, potentially overwriting epsilon budget")
            return sigma
        
        if epsilon_budget is None: raise ValueError("Either epsilon budget or must be provided")
        sigma = find_sigma_for_target_epsilon(q, total_steps,epsilon_budget, delta,verbose=self.verbose)
        if self.verbose:
            print(f"Sigma set to: {sigma}, using epsilon budget: {epsilon_budget} and delta: {delta}")
        return sigma

    def _compute_single_rdp(self,q, sigma, total_steps, lmbds):
        if total_steps <= 0: raise ValueError("Total steps must be a positive integer")
        if lmbds is None: raise ValueError("Lambdas must be provided")
        return compute_rdp_single(q, sigma, total_steps, lmbds)

    def get_spent_privacy(self,steps):
        total_rdp = self.rdp * steps
        privacy_spent, delta, _ = get_privacy_spent(self.lmbds, total_rdp, target_delta=self.delta)
        return privacy_spent

    def get_sigma(self):
        return self.sigma

    def get_clip_coeff(self):
        return self.clip_coeff

        
        



        
        