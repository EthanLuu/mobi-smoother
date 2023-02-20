class AlgoConfig:
    def __init__(self):
        self.gamma = 0.99 # discount factor
        self.k_epochs = 4 # update policy for K epochs
        self.actor_lr = 0.0003 # learning rate for actor
        self.critic_lr = 0.0003 # learning rate for critic

        self.kl_target = 0.1 # target KL divergence
        self.kl_lambda = 0.5 # lambda for KL penalty, 0.5 is the default value in the paper
        self.kl_beta = 1.5 # beta for KL penalty, 1.5 is the default value in the paper
        self.kl_alpha = 2 # alpha for KL penalty, 2 is the default value in the paper
        
        self.entropy_coef = 0.01 # entropy coefficient
        self.train_batch_size = 100 # update policy every n steps
        self.actor_hidden_dim = 128 # hidden dimension for actor
        self.critic_hidden_dim = 128  # hidden dimension for critic
