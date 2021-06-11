#coding:utf-8
#-*- coding:utf-8 -*-
import collections
import functools

import numpy as np
from context.agent_context import LinearPayoffTS
from context.agent_ht import HoeffdingTreeBandit, LinearPayoffEnsembleSampling, upperBandit
from context.agent_semipara import SemiparaBandit
from assort.comb_ban.agent_ban import ThompsonBandit
from context.env_context import *

from base.config_lib import Config
from base.experiment import BaseExperiment

# from oracle_generator.nn.model.dnn import DNN
from oracle_generator.data import FileData


def get_config():
  """Generates the config for the experiment."""
  name = 'ts-test'
  n_steps = 20000
  n_arm = 100
  dim = 2
  # context = np.random.normal(0, 0.1, (n_arm,dim))
  # linpara = (np.random.rand(dim)-0.5)*2
  delta = 0.5

  ######################### 个性化  #########################
  # n_user = 100
  # context = np.random.rand(n_user,n_arm,dim)
  # # context = (context.T/np.sqrt(np.sum(context*context,1))).T
  # context = context/np.max(np.sqrt(np.sum(context*context,2)))
  # linpara = np.random.rand(dim)
  # linpara = linpara/np.sqrt(np.sum(linpara*linpara))*(1-delta)
  # var = np.random.rand(n_arm)*delta
  # np.savez('data_user.npz', linpara=linpara, context=context, var=var)

  # data=np.load('data_user.npz')
  # context=data['context']
  # linpara=data['linpara']
  # var=data['var']
  ######################### 模拟数据  #########################
  context = np.random.rand(n_arm,dim)
  # context = (context.T/np.sqrt(np.sum(context*context,1))).T
  context = context/np.max(np.sqrt(np.sum(context*context,1)))
  linpara = np.random.rand(dim)
  linpara = linpara/np.sqrt(np.sum(linpara*linpara))*(1-delta)
  var = np.random.rand(n_arm)*delta
  np.savez('data.npz', linpara=linpara, context=context, var=var)

  data=np.load('data.npz')
  context=data['context']
  linpara=data['linpara']
  var=data['var']
  ######################### 真实数据  #########################
  # dataset = FileData('data/pool_linpayoff', feature_cols=np.array(range(4,32)), label_col=1)
  # label, raw_context = dataset.sample(n_arm)
  # context = raw_context[:, 0:dim]
  # env_context = raw_context[:, 0:dim]
  # model = torch.load('data/dnn_model_d%d'%dim)

  # ######################### 个性化 真实数据  #########################
  dataset = FileData('data/pool_context_test_s100_u100_20180515', feature_cols=np.array(range(6,6+dim)), label_col=0)
  raw_context = dataset.sample_feature_tensor(n_arm, pkey_col=1, skey_col=2)
  context = raw_context[:, :, 0:dim]
  env_context = raw_context[:, :, 0:dim]
  model = torch.load('data/dnn_model_context_d%d'%dim)
  #
  # ######################### feaProjection结果  #########################
  # projmodel = torch.load('data/tmp_feaProj_model_context_d%d' % dim)
  # projContext = projmodel.projFeat(context)
  # projAutomodel = torch.load('data/tmp_feaAutoProj_model_context_d%d' % dim)
  # projAutoContext = projAutomodel.projFeat(context)
  # projRandmodel = torch.load('data/tmp_randomProj_model_context_d%d' % dim)
  # projRandContext = projRandmodel.projFeat(context)

  agents = collections.OrderedDict(
      [
        ('tsmab2', functools.partial(ThompsonBandit, np.ones(n_arm), 1, 10.0, 1.0, 10.0)),
        # ('tsmab', functools.partial(ThompsonBandit, np.ones(n_arm), 1)),
        ('sp', functools.partial(SemiparaBandit, context, 0.3, 0.01, 0.3))
        # , ('sp2', functools.partial(SemiparaBandit, context, 0.3, 0.001, 0.3))
        # , ('sp3', functools.partial(SemiparaBandit, context, 0.3, 0.1, 0.3))
        # , ('sp5', functools.partial(SemiparaBandit, context, 1.0, 0.1, 1.0))
        # , ('sp8', functools.partial(SemiparaBandit, context, 1.0, 0.003, 1.0))
        # , ('sp7', functools.partial(SemiparaBandit, context, 5.0, 0.003, 5.0))
        # , ('sp6', functools.partial(SemiparaBandit, context, 5.0, 0.01, 5.0))
        , ('ts', functools.partial(LinearPayoffTS, context))
        # , ('ts2', functools.partial(LinearPayoffTS, context, 3.0))
        # ('ts4', functools.partial(LinearPayoffTS, context, 1.0)),
        # , ('ts3', functools.partial(LinearPayoffTS, context, 5.0))
         # ('ht', functools.partial(HoeffdingTreeBandit, context, 0.1)),
          #('tses', functools.partial(LinearPayoffEnsembleSampling, context)),
          # ('upper', functools.partial(upperBandit, context)),
          # ('proTs', functools.partial(LinearPayoffTS, projContext)),
          # ('proAutoTs', functools.partial(LinearPayoffTS, projAutoContext)),
          # ('proRandTs', functools.partial(LinearPayoffTS, projRandContext)),
       ]
  )

  environments = collections.OrderedDict(
      [
          # ('linear_payoff', functools.partial(LinearPayoffEnvironment, context, linpara))
          # ('semipara', functools.partial(SemiparaEnvironment, context, linpara, var)),
          #('linear_payoff_bernoulli', functools.partial(LinearPayoffBernoulliEnvironment, context, linpara))
          ('semipara_bernoulli', functools.partial(SemiparaBernoulliEnvironment, context, linpara, var))
          #  ('dnn_bernoulli', functools.partial(DNNBernoulliEnvironment, env_context, model))
      ]
  )
  experiments = collections.OrderedDict(
      [(name, BaseExperiment)]
  )
  n_seeds = 1
  if len(environments.keys()) == 1:
      name = '%s_%d_%d_d%f'%(environments.keys()[0], n_arm, dim, delta)
  elif len(agents.keys()) == 1:
      name = '%s_%d_%d' % (agents.keys()[0], n_arm, dim)
  config = Config(name, agents, environments, experiments, n_steps, n_seeds)
  return config