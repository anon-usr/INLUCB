""" assortment environments. """

import numpy as np

from base.environment import Environment
from base.action import Action
from base.reward import Reward
import torch
from torch import nn
from torch import autograd
from oracle_generator.data import FileData
from context.observation_context import ContextObservation


###################################################

class ContextEnvironment(Environment):

    """Environments of contextual bandit."""

    def __init__(self, context, update_reward=True):
        """Initialize the environment."""
        assert isinstance(context, np.ndarray)
        assert len(np.shape(context)) == 2 or len(np.shape(context)) == 3
        if len(np.shape(context)) == 2:
            self.context = np.array([context.copy()])
        else:
            self.context = context.copy()
        self.cnum, self.n, self.d = np.shape(self.context)

        self.cid_rand = np.random.RandomState()
        self.reward_rand = np.random.RandomState()

        self.cid = 0
        self.reward = None
        self.optimal_reward = None
        self.optimal_action = None
        self.rewards = {}
        self.t = 0

        self.cid = self._update_cid()
        if update_reward:
            self._update_reward()
            self.optimal_action = np.argsort(self.reward)[-1]
            self.optimal_reward = self.reward[self.optimal_action]
            self.rewards[self.cid]=self.reward.copy()

    def _update_cid(self):
        return self.cid_rand.randint(0, self.cnum)

    def _update_reward(self):
        """update the reward of corresponding context id"""
        pass

    def get_observation(self):
        """Returns an observation from the environment."""
        return ContextObservation(t=self.t, cid=self.cid)

    def get_optimal_reward(self):
        """Returns the optimal possible reward for the environment at that point."""
        return self.optimal_reward

    def get_expected_reward(self, action):
        """Gets the expected reward of an action."""
        assert isinstance(action, Action)
        return self.reward[action.actions[0]]

    def get_stochastic_reward(self, action):
        """Gets a stochastic reward for the action."""
        pass

    def advance(self, action, reward):
        """Updating the environment (useful for nonstationary bandit)."""
        tmp_cid = self._update_cid()
        if tmp_cid != self.cid:
            self.cid = tmp_cid
            self.reward = self.rewards.get(self.cid, None)
            if self.reward is None:
                self._update_reward()
                self.rewards[self.cid]=self.reward.copy()
            self.optimal_action = np.argsort(self.reward)[-1]
            self.optimal_reward = self.reward[self.optimal_action]
        self.t += 1

class LinearPayoffEnvironment(ContextEnvironment):
    """ time-invariant linear payoff environment"""

    def __init__(self, context, linpara, update_reward=True):
        super(LinearPayoffEnvironment, self).__init__(context, update_reward=False)
        assert isinstance(linpara, np.ndarray)
        assert len(linpara)==self.d
        self.linpara = linpara.copy()
        if update_reward:
            self._update_reward()
            self.optimal_action = np.argsort(self.reward)[-1]
            self.optimal_reward = self.reward[self.optimal_action]

    def _update_reward(self):
        self.reward = self.context[self.cid].dot(self.linpara)

    def get_stochastic_reward(self, action):
        """Gets a stochastic reward for the action.
            - first sample the purchased item and return its price
        """
        assert isinstance(action, Action)
        rval = self.reward[action.actions]+self.reward_rand.standard_normal()*0.1
        srwd = Reward(total_reward=np.sum(rval), rewards=rval)
        return srwd

class SemiparaEnvironment(LinearPayoffEnvironment):
    def __init__(self, context, linpara, var=None, update_reward=True):
        super(SemiparaEnvironment, self).__init__(context, linpara, update_reward=False)
        assert var is None or (isinstance(var, np.ndarray) and (len(var.shape)==1 or len(var.shape)==2))

        if var is None:
            self.var = np.zeros((self.cnum, self.n))
        elif len(var.shape)==1:
            self.var = np.array([var.copy()])
        else:
            self.var = var.copy()

        assert self.var.shape[0]==self.cnum and self.var.shape[1]==self.n
        if update_reward:
            self._update_reward()
            self.optimal_action = np.argsort(self.reward)[-1]
            self.optimal_reward = self.reward[self.optimal_action]

    def _update_reward(self):
        self.reward = (self.context[self.cid]).dot(self.linpara)+self.var[self.cid]

class LinearPayoffBernoulliEnvironment(LinearPayoffEnvironment):
    def __init__(self, context, linpara):
        super(LinearPayoffBernoulliEnvironment, self).__init__(context, linpara)
        assert np.all(self.reward >= 0.0) and np.all(self.reward <= 1.0)

    def get_stochastic_reward(self, action):
        """Gets a stochastic reward for the action.
            - first sample the purchased item and return its price
        """
        assert isinstance(action, Action)
        rval = self.reward_rand.binomial(1, self.reward[action.actions])
        srwd = Reward(total_reward=np.sum(rval), rewards=rval)
        return srwd


class SemiparaBernoulliEnvironment(SemiparaEnvironment):
    def __init__(self, context, linpara, var=None):
        super(SemiparaBernoulliEnvironment, self).__init__(context, linpara, var=var)
        assert np.all(self.reward >= 0.0) and np.all(self.reward <= 1.0)

    def get_stochastic_reward(self, action):
        """Gets a stochastic reward for the action.
            - first sample the purchased item and return its price
        """
        assert isinstance(action, Action)
        rval = self.reward_rand.binomial(1, self.reward[action.actions])
        srwd = Reward(total_reward=np.sum(rval), rewards=rval)
        return srwd

class DNNBernoulliEnvironment(ContextEnvironment):
    def __init__(self, context, model, update_reward=True):
        super(DNNBernoulliEnvironment, self).__init__(context, update_reward=False)
        assert isinstance(model, nn.Module)
        self.model = model
        if update_reward:
            self._update_reward()
            self.optimal_action = np.argsort(self.reward)[-1]
            self.optimal_reward = self.reward[self.optimal_action]

    def _update_reward(self):
        tensor = torch.FloatTensor(self.context[self.cid])
        out = self.model.predict(autograd.Variable(tensor))
        self.reward = out.data.numpy().reshape(-1)
        # self.optimal_action = np.argsort(self.reward)[-1]
        # self.optimal_reward = self.reward[self.optimal_action]

    def get_stochastic_reward(self, action):
        """Gets a stochastic reward for the action.
            - first sample the purchased item and return its price
        """
        assert isinstance(action, Action)
        rval = self.reward_rand.binomial(1, self.reward[action.actions])
        srwd = Reward(total_reward=np.sum(rval), rewards=rval)
        return srwd