import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Parsing_Net(nn.Module):
	def __init__(self, 
				 dim_in,
				 dim_hid,
				 n_slots=5,
				 n_lookback=1,
				 resolution=0.1,
				 dropout=0.5):
		super(Parsing_Net, self).__init__()

		self.dim_in = dim_in
		self.dim_hid = dim_hid
		self.n_slots = n_slots
		self.n_lookback = n_lookback
		self.resolution = resolution

		self.Dropout = nn.Dropout(dropout)

		self.layer = nn.Sequential(nn.Dropout(dropout),
								   nn.Conv1d(dim_in, dim_hid, (n_lookback + 1)),
								   nn.BatchNorm1d(dim_hid),
								   nn.ReLU(),
								   nn.Dropout(dropout),
								   nn.Conv1d(dim_hid, 2, 1, groups=2).
								   nn.sigmoid())

	def forward(self, emb, parser_state):
		emb_last, cum_gate = parser_state
		ntimestep = emb.size(0)

		emb_last = torch.cat([emb_last, emb], dim=0)
		emb = emb_last.transpose(0, 1).transpose(1, 2)
