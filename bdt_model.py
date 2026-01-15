import numpy as np

class BayesianDecisionModel:
    def __init__(self):
        self.classes = None
        self.prior = {}
        self.mean = {}
        self.var = {}
        self.loss = {}

    def posterior(self, x):
        post = {}
        for c in self.classes:
            loglik = -0.5*np.sum(np.log(2*np.pi*self.var[c]) + ((x-self.mean[c])**2)/self.var[c])
            post[c] = np.exp(loglik) * self.prior[c]
        s = sum(post.values())
        return {k:v/s for k,v in post.items()}

    def clinical_decision(self, posterior):
        return min(self.loss, key=lambda d: sum(self.loss[d][c]*posterior[c] for c in self.classes))
