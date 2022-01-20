import numpy as np                                                                                                                    
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.nn.functional import one_hot
from abc import abstractmethod, ABC 
import tqdm
import matplotlib.pyplot as plt
from functools import partial
import os
import argparse
from functools import partial



class FFN(nn.Module):

    def __init__(self, sizes, activation=nn.ReLU, output_activation=nn.Identity):
        super().__init__()

        layers = []
        layers.append(nn.BatchNorm1d(sizes[0]))
        for j in range(1,len(sizes)):
            layers.append(nn.Linear(sizes[j-1], sizes[j]))
            if j<(len(sizes)-1):
                layers.append(nn.BatchNorm1d(sizes[j]))
                layers.append(activation())
            else:
                #layers.append(nn.BatchNorm1d(sizes[j]))
                layers.append(output_activation())

        self.net = nn.Sequential(*layers)

    def forward(self, *args):
        x = torch.cat(args, -1)
    
        return self.net(x)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(m.bias)#m.bias.zero_()#, gain=nn.init.calculate_gain('relu'))



def U(x, a):
    """ Utility function
    Parameters
    ----------
    a: float
        Utility parameter
    x: torch.Tensor
        Tensor of shape (batch_size, 1)
    """
    assert a>0, 'a needs to be positive'
    output = torch.zeros_like(x)
    output[x.lt(0)] = x[x.lt(0)]
    output[x.ge(0)] = 1-torch.exp(-a*x[x.ge(0)])
    return output


def Brownian_bridge_step(b: torch.Tensor, t: float, T: float, tau: float, **kwargs):
    """
    Step of Brownian bridge given by SDE
    dS_t = (b-S_t)/(T-t)dt + dW_t , S_0 = a 

    Parameters
    ----------
    t: float
        current time 
    T: float
        final time 
    tau: float
        time discretisation step size
    b: torch.Tensor
        Tensor of shape (batch_size, *). Samples of S_t at current time t
    """
    
    next_b = b + (-b)/(T-t) * tau + np.sqrt(tau) * torch.randn_like(b)
    return next_b


def GBM_step(sigma: float, tau: float, s: torch.Tensor, **kwargs):
    """
    Step of GBM with zero drift
    """
    if kwargs['increment_type'] == "Brownian Bridge":
        s_next = s * torch.exp(-0.5 * sigma**2 * tau + sigma * kwargs['dW'])
    else:
        s_next = s * torch.exp(-0.5 * sigma**2 * tau + sigma * np.sqrt(tau)*torch.randn_like(s)) 
    return s_next


class ActorCritic():
    
    def __init__(self, a: float, gamma: float, kappa: float, discount_factor: float, device = 'cpu', **kwargs):

        """
        
        Parameters
        ----------
        discount_factor: float
            Discount factor in Bellman equation between (0,1)
        tau: float
            Change of time, in time discretisation
        sigma: float
            Diffusion in LQR SDE. I assume the diffusion is constant. Can be easily changed
        gamma: float
            (1-gmma) is fee
        kappa: float
            slippage        
        device: str
            Device where things are run
        
        
        """
        self.d = 3 # dimension of the state (s, r_alpha, r_beta)
        self.device=device
                
        # Running cost 
        self.f = partial(U, a=a)
        
        # fee
        self.gamma = gamma
        
        # slippage
        self.kappa = kappa
        
        # policy
        self.alpha = FFN(sizes = [self.d + 1] + kwargs['hidden_dims'] + [2], output_activation=nn.Softplus).to(device) # Soft policy. input of alpha is (s,r^alpha, r^beta,z) where z is input noise
        self.C = FFN(sizes = [self.d] + kwargs['hidden_dims'] + [3], output_activation=nn.Softmax).to(device) # if C=0 sell alpha. if C=1 sell beta
        self.alpha.apply(init_weights)
        self.optimizer_alpha = torch.optim.Adam(list(self.alpha.parameters()) + list(self.C.parameters()), lr=0.005)
        self.scheduler_alpha = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_alpha, milestones=[3000,4000], gamma=0.1)
        
        # value function
        self.v = FFN(sizes = [self.d] + kwargs['hidden_dims'] + [1]).to(device) # input of v is x
        self.v.apply(init_weights)
        self.optimizer_v = torch.optim.Adam(self.v.parameters(), lr=0.005)
        self.scheduler_v = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_v, milestones=[3000,4000], gamma=0.1)
        
        # discount factor for Bellman equation
        self.discount_factor = discount_factor
        
        
    
    def step(self, x: torch.Tensor, n_mc: int, train: bool = True, **kwargs):
        """

        Parameters
        ----------
        x: torch.Tensor
            State
        n_mc: int
            Monte Carlo samples to approximate explorative running_cost
        train: bool
            boolean that tells whether we are training, or whether we are sampling from the soft policy
        kwargs: dict
            arguments necessary to calculate a step of mid-price 
        """
        n_batch = x.shape[0]
        # MONTE CARLO!
        x_mc = torch.repeat_interleave(x, n_mc, dim=0) # (n_mc * n_batch, d)
        z_a = 2*torch.randn(n_mc*n_batch, 1, device=x.device) # input noise for the policy generator
        a = self.alpha(x_mc, z_a)
        c = self.C(x_mc)#.reshape(-1)
        
        # Environment step
        s, r_alpha, r_beta = x_mc[:,0], x_mc[:,1], x_mc[:,2] 
        Delta_alpha, Delta_beta = a[:,0], a[:,1]
        
        if not train:
            m = Categorical(c)
            c = one_hot(m.sample(), num_classes = 3)
            Delta_alpha[c[:,1].eq(0.)] = 0
            Delta_beta[c[:,0].eq(0.)] = 0
        
        #z_s = torch.randn_like(s) # (n_mc*n_batch, device=x.device)
        if kwargs.get('increment_type') == 'Brownian Bridge':
            b_next = Brownian_bridge_step(**kwargs)
            kwargs['dW'] = b_next.clone() - kwargs['b'].clone()
            kwargs['b'] = b_next.clone()

        s_step = partial(GBM_step, **kwargs)
        s_next = s_step(s=s) #s * torch.exp(-0.5 * self.sigma**2 * self.tau + self.sigma * np.sqrt(self.tau)*z_s) # (n_mc*n_batch)
        
        r_alpha_next = c[:,0] * r_alpha*r_beta / (r_beta + self.gamma*Delta_beta) + c[:,1] * (r_alpha + Delta_alpha) + c[:,2] * r_alpha
        r_beta_next = c[:,0] * (r_beta + Delta_beta) + c[:,1] * r_alpha*r_beta / (r_alpha + self.gamma*Delta_alpha) + c[:,2] * r_beta

        profit_alpha = c[:,0] * (r_alpha - r_alpha*r_beta / (r_beta + self.gamma*Delta_beta) - Delta_beta*(s-self.kappa*Delta_beta))
        Delta_beta_const_market = r_beta - r_alpha*r_beta/(r_alpha + self.gamma*Delta_alpha)
        profit_alpha += c[:,1] * (Delta_beta_const_market * (s - self.kappa*Delta_beta_const_market) - Delta_alpha)
        running_cost = self.f(profit_alpha).reshape(-1,1)
       
        x_next = torch.stack((s_next, r_alpha_next, r_beta_next), dim=1)
        
        return torch.cat([a, c],1), x_next, running_cost, profit_alpha

    
    
    
    def _dynamic_programming(self, x: torch.Tensor, n_mc: int, **kwargs):
        """
        Performs one step environment step and return bellman loss
        
        Parameters
        ----------
        x: torch.Tensor
            tensor. tensor of shape (N_batch, 3)
        n_mc: int
            Number of monte carlo samples to approximate drift
        kwargs: dict
            arguments necessary to calculate a step of mid-price 
        
        Returns
        ------
        bellman_loss: torch.Tensor
            bellman loss: ( v(x) - 1/N_mc \sum(f + delta * v(x_next)) )^2
        
        bellman_approx: torch.Tensor
            bellman approximation of v(x): 1/N_mc \sum(f + delta * v(x_next))
        
        """
        n_batch = x.shape[0]
        _, x_next, running_cost, _ = self.step(x, n_mc, train=True, **kwargs)
        
        # bellman approx
        bellman_approx = running_cost + self.discount_factor * self.v(x_next) # (n_batch*n_mc, 1)
        bellman_approx = bellman_approx.reshape(n_batch, n_mc, -1).mean(1) # (n_batch, 1)
        
        # bellman loss
        bellman_loss = torch.pow(self.v(x) - bellman_approx.detach(),2).mean()
        return bellman_loss, bellman_approx.mean()
    
    def update_alpha(self, n_batch, n_mc, **kwargs):
        """
        Gradient ascent on alpha to maximise bellman approx

        Parameters
        ----------
        n_batch: int
            batch size
        n_mc: int
            Monte Carlo size for Monte Carlo approximation of running cost to have some exploration
        kwargs: dict
            arguments necessary to calculate a step of mid-price 
        """
        
        toggle(self.v, to=False)
        toggle(self.alpha, to=True)
        toggle(self.C, to=True)
        #self.v.eval()
        self.alpha.train()
        self.C.train()
        
        #x0 = torch.randn(n_batch, self.d)
        x0 = sample_x0(n_batch, self.d, device = self.device)
        self.optimizer_alpha.zero_grad()
        _, bellman_approx = self._dynamic_programming(x0, n_mc, **kwargs)
        bellman_approx = -1 * bellman_approx # we want to maximise!
        bellman_approx.backward()
        self.optimizer_alpha.step()
        self.scheduler_alpha.step()
        return -bellman_approx.detach()
    
    def update_v(self, n_batch, n_mc, **kwargs):
        """
        Gradient descent on to minimise bellman loss
        
        Parameters
        ----------
        n_batch: int
            batch size
        n_mc: int
            Monte Carlo size for Monte Carlo approximation of running cost to have some exploration
        kwargs: dict
            arguments necessary to calculate a step of mid-price 
        """
        
        toggle(self.v, to=True)
        toggle(self.alpha, to=False)
        toggle(self.C, to=False)
        self.v.train()
        
        x0 = sample_x0(n_batch, self.d, device=self.device)
        self.optimizer_v.zero_grad()
        bellman_loss, _ = self._dynamic_programming(x0, n_mc, **kwargs)
        bellman_loss.backward()
        self.optimizer_v.step()
        self.scheduler_v.step()
        return bellman_loss.detach()


def sample_x0(batch_size, dim, device='cpu', ):
    sigma = 0.3
    mu = np.log(10000.0)
    z = torch.randn(batch_size, dim, device=device)
    x0 = torch.exp((mu-0.5*sigma**2) + sigma*z) # lognormal
    x0[:,1] = x0[:,1] * 10 
    x0[:,0] = torch.clamp(x0[:,1]/x0[:,2] + 5*torch.randn_like(x0[:,0]), min=5.)
    #x0 = -10 + 20*torch.rand(batch_size, dim, device=device)
    return x0

# freeze / unfreeze networks' parameters
def toggle(net: nn.Module,  to: bool):
    for p in net.parameters():
        p.requires_grad_(to)


def make_plots(agent, path_results, n_mc, render=False, **kwargs):
    """
    kwargs: dict
        arguments for s process
    """

    agent.v.eval()
    agent.alpha.eval()
    agent.C.eval()
    x0 = sample_x0(n_mc, 3, device=device)
    n_steps = 100
    
    if kwargs['increment_type'] == "Brownian Bridge":
        kwargs.update(dict(b=torch.zeros_like(x0[:,0]), T=kwargs['tau']*n_steps))

    x0[:,0] = x0[:,1] / x0[:,2]
    #x0[:,2] += 50*torch.randn_like(x0[:,2]) 
    #x0[:,1] += 50*torch.randn_like(x0[:,1]) 
    path = [x0]
    actions = []
    PnL = []
    utility = []
    
    for i in range(1, n_steps):
        kwargs['t'] = i * kwargs['tau']
        with torch.no_grad():
            a, x, u, profit_alpha = agent.step(path[-1], n_mc=1, train=False, **kwargs)
        PnL.append(profit_alpha)
        path.append(x)
        actions.append(a)
        utility.append(u.reshape(-1))
    path = torch.stack(path, 1)
    s, r_alpha, r_beta = path[...,0].cpu(), path[...,1].cpu(), path[...,2].cpu() # (n_mc, L)
    V = r_alpha + s*r_beta # (n_mc, L)
    actions = torch.stack(actions, 1) # (n_mc, L, 3)
    PnL = torch.stack(PnL,1) # (n_mc, L)
    utility = torch.stack(utility, 1) # (n_mc, L)
    c = actions[...,-3:]
    
    if render and n_mc<=10:
        for i in range(n_mc):
            print(i)
            fig, ax = plt.subplots(1,3, figsize=(12,4))
            ax[0].plot(actions[i,:,0].cpu().numpy(), '.--', label=r'$\Delta^\alpha$')
            ax[0].plot(actions[i,:,1].cpu().numpy(), '.--', label=r'$\Delta^\beta$')
            ax[0].set_title('Action')
            ax[0].set_xlabel('Step')
            ax[0].set_ylabel('Amount')
            ax[0].legend()
            ax[0].grid()
            ax[1].plot(PnL[i].cumsum(0).cpu())
            ax[1].set_title('Cumulative PnL')
            ax[1].set_xlabel('Step')
            ax[1].set_ylabel(r'Profit in $\alpha$')
            ax[1].grid()
            ax[2].plot(r_alpha[i]/r_beta[i], label=r'$R^\alpha / R^\beta$')
            ax[2].plot(s[i], label=r'$S$')
            ax[2].set_title(r'$(S_t, R_t^\alpha / R^\beta)$ evolution')
            ax[2].set_xlabel('Step')
            ax[2].set_ylabel('Amount')
            ax[2].legend()
            ax[2].grid()
            fig.tight_layout()
            fig.savefig(os.path.join(path_results, 'trajectories_{}.pdf'.format(i)))
            plt.close()
    
    return (V[:,-1]-V[:,0]).numpy() # n_mc




if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', action='store_true', default=False)
    args = parser.parse_args()

    results_path = './numerical_results'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    
    a = 0.005 # utility function parameter
    gamma = 0.99 # (1-gamma) being fee
    kappa = 0.001 # slippage
    sigma = 0.3 # diffusion of GBM                                                                                                            
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    discount_factor = 0.8
    tau = 0.1 # time discretisation step length

    # learning hyperparameters
    n_batch = 800
    n_mc = 200
    hidden_dims = [40,40]
    n_updates_alpha = 10
    n_updates_v = 10
    max_updates = 10000

    agent = ActorCritic(a=a, gamma=gamma, kappa=kappa, discount_factor=discount_factor, hidden_dims=hidden_dims, device=device)
    if args.evaluate:
        state = torch.load(os.path.join(results_path, 'state.pth.tar'), map_location=device)
        agent.v.load_state_dict(state['v'])
        agent.alpha.load_state_dict(state['alpha'])
        agent.C.load_state_dict(state['C'])

        #mc = 10
        #s_args = dict(sigma=sigma, tau=tau, increment_type="Brownian Bridge")
        #make_plots(agent, results_path, mc, **s_args)
        n_sigmas = 10 
        n_gammas = 10
        mc = 150000
        sigmas = np.linspace(0.10,0.40,n_sigmas)
        gammas = np.linspace(0.9,0.999, n_gammas)
        heatmap = np.zeros((n_sigmas,n_gammas,mc))
        for id_sigma, sigma_ in enumerate(sigmas):
            s_args = dict(sigma=sigma_, tau=tau, increment_type="Brownian Bridge")
            for id_gamma, gamma_ in enumerate(gammas):
                print('gamma={:.3f}, sigma={:.3f}'.format(gamma_, sigma_))
                agent.gamma = gamma_
                results_path_gamma_ = os.path.join(results_path, 'sigma_{:.3f}'.format(sigma_), 'gamma_{:.3f}'.format(gamma_))
                if not os.path.exists(results_path_gamma_):
                    os.makedirs(results_path_gamma_)
                MM_diff = make_plots(agent, results_path_gamma_, mc, **s_args)
                heatmap[id_sigma, id_gamma] = np.array(MM_diff)
        
        fig, ax = plt.subplots()
        im = ax.imshow(heatmap.mean(2))
        ax.set_xticks(np.arange(n_gammas))
        ax.set_xticklabels(labels=['{:.2f}'.format(gamma_) for gamma_ in gammas])
        ax.set_yticks(np.arange(n_sigmas))
        ax.set_yticklabels(labels=['{:.2f}'.format(sigma_) for sigma_ in sigmas])
        ax.set_xlabel(r'$\gamma$')
        ax.set_ylabel(r'$\sigma$')
        ax.set_title(r'$E [V_T - V_0]$')
        plt.colorbar(im, ax=ax)
        fig.savefig(os.path.join(results_path, 'heatmap_mean.pdf'))
        
        fig, ax = plt.subplots()
        im = ax.imshow(heatmap.std(2))
        ax.set_xticks(np.arange(n_gammas))
        ax.set_xticklabels(labels=['{:.2f}'.format(gamma_) for gamma_ in gammas])
        ax.set_yticks(np.arange(n_sigmas))
        ax.set_yticklabels(labels=['{:.2f}'.format(sigma_) for sigma_ in sigmas])
        ax.set_xlabel(r'$\gamma$')
        ax.set_ylabel(r'$\sigma$')
        ax.set_title(r'$(Var [V_T - V_0])^{1/2}$')
        plt.colorbar(im, ax=ax)
        fig.savefig(os.path.join(results_path, 'heatmap_std.pdf'))

    else:
        s_args = dict(sigma=sigma, tau=tau, increment_type="BM")
        # learning
        pbar = tqdm.tqdm(total = max_updates)
        count = 0
        bellman_loss = []
        bellman_approx = []
        while count < max_updates:
                
            for i in range(n_updates_v):
                loss = agent.update_v(n_batch = n_batch, n_mc = n_mc, **s_args)
                bellman_loss.append(loss)
                pbar.write('bellman loss: {:1.2e}'.format(loss.item()))
            
            for i in range(n_updates_alpha):
                v_approx = agent.update_alpha(n_batch = n_batch, n_mc = n_mc, **s_args)
                bellman_approx.append(v_approx)
                pbar.write('bellman approx: {:1.2e}'.format(v_approx.item()))
                
            count += n_updates_alpha + n_updates_v
            pbar.update(n_updates_alpha + n_updates_v)
        
        # save results
        state = {"alpha":agent.alpha.state_dict(), "v":agent.v.state_dict(), "C": agent.C.state_dict()}
        torch.save(state, os.path.join(results_path, "state.pth.tar"))
        # plots
        filename = os.path.join(results_path, 'trajectories.pdf')
        make_plots(agent, results_path, n_mc=10, render=True, **s_args)
