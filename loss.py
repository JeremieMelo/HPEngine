import torch
x = torch.ones(8)
y = torch.ones(8)
size_x = torch.ones(8)
size_y = torch.ones(8)

num_nodes = pos.numel() // 2
x_overlap, y_overlap = 0, 0
for i in range(self.num_movable_nodes):
    for j in range(i, self.num_movable_nodes):
        x_overlap += max(min(pos[i] + self.node_size_x[i], pos[j] + self.node_size_x[j]) - max(pos[i], pos[j]), 0)
        y_overlap += max(min(pos[num_nodes + i] + self.node_size_y[i], pos[num_nodes + j] + self.node_size_y[j]) - max(pos[num_nodes + i], pos[num_nodes + i]), 0)

pos_x1 = pos[:self.num_movable_nodes]
pos_y1 = pos[num_nodes:num_nodes+self.num_movable_nodes]
pos_x2 = pos_x1 + self.node_size_x[:self.num_movable_nodes]
pos_y2 = pos_y1 + self.node_size_y[:self.num_movable_nodes]

pos_x1 = pos_x1.unsqueeze(0)
pos_x11 = pos_x1.unsqueeze(-1)
pos_x2 = pos_x2.unsqueeze(0)
pos_x21 = pos_x2.unsqueeze(-1)
x_overlap = (F.relu(torch.min(pos_x2, pos_x21) - torch.max(pos_x1, pos_x11)).view(-1)[torch.arange(0, self.num_movable_nodes**2, self.num_movable_nodes+1)].zero_()**2).sum()
del pos_x11
del pos_x21
pos_y1 = pos_y1.unsqueeze(0)
pos_y11 = pos_y1.unsqueeze(-1)
pos_y2 = pos_y2.unsqueeze(0)
pos_y21 = pos_y2.unsqueeze(-1)
y_overlap = (F.relu(torch.min(pos_y2, pos_y21) - torch.max(pos_y1, pos_y11)).view(-1)[torch.arange(0, self.num_movable_nodes**2, self.num_movable_nodes+1)].zero_()**2).sum()

pos_x1 = pos[:self.num_movable_nodes]
pos_y1 = pos[num_nodes:num_nodes+self.num_movable_nodes]
pos_x2 = pos_x1 + self.node_size_x[:self.num_movable_nodes]
pos_y2 = pos_y1 + self.node_size_y[:self.num_movable_nodes]
pos_x1 = pos_x1.unsqueeze(0)
pos_x11 = pos_x1.unsqueeze(-1)
pos_x2 = pos_x2.unsqueeze(0)
pos_x21 = pos_x2.unsqueeze(-1)
total_x_overlap = 0
tile_size = self.num_movable_nodes//64
for i in range(0, self.num_movable_nodes, tile_size):
    x_overlap = F.relu(torch.min(pos_x2, pos_x21[i:i+tile_size,:]) - torch.max(pos_x1, pos_x11[i:i+tile_size,:])).view(-1)
    x_overlap[torch.arange(0, x_overlap.numel(), self.num_movable_nodes+1)].zero_()
    total_x_overlap += torch.dot(x_overlap, x_overlap)

del pos_x1
del pos_x11
del pos_x2
del pos_x21
pos_y1 = pos_y1.unsqueeze(0)
pos_y11 = pos_y1.unsqueeze(-1)
pos_y2 = pos_y2.unsqueeze(0)
pos_y21 = pos_y2.unsqueeze(-1)
total_y_overlap = 0
for i in range(0, self.num_movable_nodes, tile_size):
    y_overlap = F.relu(torch.min(pos_y2, pos_y21[i:i+tile_size,:]) - torch.max(pos_y1, pos_y11[i:i+tile_size,:])).view(-1)
    y_overlap[torch.arange(0, y_overlap.numel(), self.num_movable_nodes+1)].zero_()
    total_y_overlap += torch.dot(y_overlap, y_overlap)
