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
								   nn.Conv1d(dim_hid, 2, 1, groups=2),
								   nn.Sigmoid())

	def forward(self, emb, parser_state):
		emb_last, cum_gate = parser_state
		ntimestep = emb.size(0)

		emb_last = torch.cat([emb_last, emb], dim=0)
		emb = emb_last.transpose(0, 1).transpose(1, 2)

		gates = self.layer(emb)
		gate = gates[:, 0, :]
		gate_next = gates[:, 1, :]
		cum_gate = torch.cat([cum_gate, gate], dim=1)
		gate_hat = torch.stack([cum_gate[:, i:i + ntimestep] for i in range(self.nslots, 0, -1)], dim=2)

		if self.hard:
			memory_gate = (F.hardtanh((gate[:, :, None] - gate_hat) / self.resolution * 2 + 1) + 1) / 2
		else:
			memory_gate = F.sigmoid((gate[:, :, None] - gate_hat) / self.resolution * 10 + 5)
		memory_gate = torch.cumprod(memory_gate, dim=2)
		memory_gate = torch.unbind(memory_gate, dim=1)

		if self.hard:
			memory_gate_next = (F.hardtanh((gate_next[:, :, None] - gate_hat) / self.resolution * 2 + 1) + 1) / 2
		else:
			memory_gate_next = F.sigmoid((gate_next[:, :, None] - gate_hat) / self.resolution * 10 + 5)
		memory_gate_next = torch.cumprod(memory_gate_next, dim=2)
		memory_gate_next = torch.unbind(memory_gate_next, dim=1)

		return (memory_gate, memory_gate_next), gate, (emb_last[-self.n_lookback:], cum_gate[:, -self.n_slots:])

	def init_hidden(self, batch_size):
		weight = next(self.parameters()).data
		self.ones = weight.new(batch_size, 1).zero_() + 1
		return weight.new(self.n_lookback, batch_size, self.dim_in).zero_(), \
			   weight.new(batch_size, self.n_slots).zero_() + numpy.inf

def test():
	pmodel = Parsing_Net(dim_in=100,
						 dim_hid=64,
						 n_slots=5,
						 n_lookback=1,
						 resolution=0.1,
						 dropout=0.5)
	a, b = pmodel.init_hidden(100)


if __name__ == "__main__":
	test()
