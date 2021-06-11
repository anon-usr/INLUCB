import numpy as np

from base.agent import Agent

from base.action import Action
from base.reward import Reward


class SemiparaBandit(Agent):
    def __init__(self, context, sigma1=0.3, sigma2=0.01, sigma3=0.3):
        """Initialize the agent."""
        assert isinstance(context, np.ndarray)
        assert len(np.shape(context)) == 2
        self.n, self.d = np.shape(context)
        self.context = context.copy()
        self.sigma1 = sigma1#*sigma1
        self.sigma2 = sigma2#*sigma2
        self.sigma3 = sigma3#*sigma3

        self.A = np.eye(self.d, self.d)/self.sigma3
        self.b = np.zeros(self.d)/self.sigma3
        self.invA = np.linalg.inv(self.A)
        self.linpara = self.invA.dot(self.b)
        self.gamma = np.zeros(self.n)

        self.reward = np.zeros(self.n)
        self.exposure = np.zeros(self.n)

        self.theta_rand = np.random.RandomState()
        self.gamma_rand = np.random.RandomState()

    def update_observation(self, observation, action, reward):
        """Add an observation to the records."""
        assert isinstance(action, Action)
        assert isinstance(reward, Reward)
        pre_aw = self.exposure[action.actions] / (self.sigma1 + self.exposure[action.actions] * self.sigma2)
        pre_bw = self.reward[action.actions]/(self.sigma1 + self.exposure[action.actions]*self.sigma2)
        self.reward[action.actions] += reward.rewards
        self.exposure[action.actions] += 1
        cur_aw = self.exposure[action.actions] / (self.sigma1 + self.exposure[action.actions] * self.sigma2)
        cur_bw = self.reward[action.actions]/(self.sigma1 + self.exposure[action.actions]*self.sigma2)
        delta_aw = cur_aw-pre_aw
        delta_bw = cur_bw-pre_bw
        xmat = self.context[action.actions].reshape(1, -1)
        self.A += (xmat.T*delta_aw).dot(xmat)
        self.b += self.context[action.actions].T.dot(delta_bw)
        self.invA = np.linalg.inv(self.A)
        self.linpara = self.invA.dot(self.b)

    def pick_action(self, observation):
        """Select an action based upon the policy + observation."""
        # self.A = np.eye(self.d, self.d) + self.context.T.dot(self.context) / self.sigma2
        # self.invA = np.linalg.inv(self.A)
        # for l in range(10):
        #     self.b = self.gamma.dot(self.context)/self.sigma2
        #     self.linpara = self.invA.dot(self.b)
        #     self.smppara = np.random.multivariate_normal(self.linpara, self.var * self.var * self.invA)
        #     self.gamma = np.random.normal((self.sigma2*self.reward+self.sigma1*self.smppara.dot(self.context.T))/(self.exposure*self.sigma2+self.sigma1), self.sigma1*self.sigma2/(self.exposure*self.sigma2+self.sigma1))

        self.smppara = self.theta_rand.multivariate_normal(self.linpara, self.invA)
        pri_gamma = self.smppara.dot(self.context.T)
        mean_gamma = (self.sigma2 * self.reward + self.sigma1 * pri_gamma) / (self.exposure * self.sigma2 + self.sigma1)
        gamma_cov = self.sigma1 * self.sigma2 / (self.sigma1 + self.exposure * self.sigma2)
        self.gamma = self.gamma_rand.normal(mean_gamma, gamma_cov)
        optimal_action = Action(actions=np.argsort(self.gamma)[-1])

        return optimal_action

class SemiparaContextBandit(Agent):
    def __init__(self, context, sigma1=0.3, sigma2=0.01, sigma3=0.3):
        """Initialize the agent."""
        assert isinstance(context, np.ndarray)
        assert len(np.shape(context)) == 2 or len(np.shape(context)) == 3
        if len(np.shape(context)) == 2:
            self.context = np.array([context.copy()])
        else:
            self.context = context.copy()
        self.cnum, self.n, self.d = np.shape(self.context)

        self.A = np.eye(self.d, self.d)
        self.b = np.zeros(self.d)
        self.invA = np.linalg.inv(self.A)
        self.linpara = self.invA.dot(self.b)
        self.As = []
        self.bs = []
        self.invAs = []
        self.linparas = []
        for i in range(self.n):
            tmp_A = np.eye(self.d, self.d)
            tmp_b = np.zeros(self.d)
            tmp_invA = np.linalg.inv(tmp_A)
            self.As.append(tmp_A)
            self.bs.append(tmp_b)
            self.invAs.append(tmp_invA)
            self.linparas.append(tmp_invA.dot(tmp_b))

        self.reward = np.zeros(self.n)

        self.theta_rand = np.random.RandomState()
        self.thetas_rand = np.random.RandomState()

    def update_observation(self, observation, action, reward):
        """Add an observation to the records."""
        assert isinstance(action, Action)
        assert isinstance(reward, Reward)
        pre_aw = self.exposure[action.actions] / (self.sigma1 + self.exposure[action.actions] * self.sigma2)
        pre_bw = self.reward[action.actions]/(self.sigma1 + self.exposure[action.actions]*self.sigma2)
        self.reward[action.actions] += reward.rewards
        self.exposure[action.actions] += 1
        cur_aw = self.exposure[action.actions] / (self.sigma1 + self.exposure[action.actions] * self.sigma2)
        cur_bw = self.reward[action.actions]/(self.sigma1 + self.exposure[action.actions]*self.sigma2)
        delta_aw = cur_aw-pre_aw
        delta_bw = cur_bw-pre_bw
        xmat = self.context[action.actions].reshape(1, -1)
        self.A += (xmat.T*delta_aw).dot(xmat)
        self.b += self.context[action.actions].T.dot(delta_bw)
        self.invA = np.linalg.inv(self.A)
        self.linpara = self.invA.dot(self.b)

    def pick_action(self, observation):
        """Select an action based upon the policy + observation."""
        # self.A = np.eye(self.d, self.d) + self.context.T.dot(self.context) / self.sigma2
        # self.invA = np.linalg.inv(self.A)
        # for l in range(10):
        #     self.b = self.gamma.dot(self.context)/self.sigma2
        #     self.linpara = self.invA.dot(self.b)
        #     self.smppara = np.random.multivariate_normal(self.linpara, self.var * self.var * self.invA)
        #     self.gamma = np.random.normal((self.sigma2*self.reward+self.sigma1*self.smppara.dot(self.context.T))/(self.exposure*self.sigma2+self.sigma1), self.sigma1*self.sigma2/(self.exposure*self.sigma2+self.sigma1))

        self.smppara = self.theta_rand.multivariate_normal(self.linpara, self.invA)
        pri_gamma = self.smppara.dot(self.context.T)
        mean_gamma = (self.sigma2 * self.reward + self.sigma1 * pri_gamma) / (self.exposure * self.sigma2 + self.sigma1)
        gamma_cov = self.sigma1 * self.sigma2 / (self.sigma1 + self.exposure * self.sigma2)
        self.gamma = self.gamma_rand.normal(mean_gamma, gamma_cov)
        optimal_action = Action(actions=np.argsort(self.gamma)[-1])

        return optimal_action