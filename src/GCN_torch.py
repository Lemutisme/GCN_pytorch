# -*- coding: utf-8 -*-
# wroten by DZ
import pandas as pd

import dgl
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.animation as animation
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# set args
epoches = 20
hidden_size = 8
country = "United States"
torch.cuda.set_device(0) # if you don't have a cuda supported gpu, please Comments this line

# load data
names_flights = ('airline,airline_id,''source,source_id,''dest,dest_id,''codeshare,stops,equipment').split(',')
names_airports = ('id,name,city,country,iata,icao,lat,lon,''alt,timezone,dst,tz,type,source').split(',')
flights_all = pd.read_csv('./Data/routes.dat',names=names_flights,header=None)
airports_all = pd.read_csv('./Data/airports.dat', header=None, names=names_airports,na_values='\\N')

# use networkx & dgl to build graph
def get_network(country = "all"):
    if country == "all":
        airports = airports_all
    else:
        airports = airports_all[airports_all['country'] == country]
    airports = airports[airports['iata'].notna()]
    airports = airports.set_index('iata')

    flights = flights_all[flights_all['source'].isin(airports.index) & flights_all['dest'].isin(airports.index)]
    edges = flights[['source', 'dest']].values
    g = nx.from_edgelist(edges)
    pos = {airport: (v['lon'], v['lat']) for airport, v in airports.to_dict('index').items()}
    return g, pos

# build GCN model
def gcn_message(edges):
    return {"msg": edges.src["h"]}

def gcn_reduce(nodes):
    return {"h": torch.sum(nodes.mailbox["msg"], dim=1)}

class GCNLayer(nn.Module):

    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, inputs):
        # g is the graph and the inputs is the input node features
        g.ndata["h"] = inputs
        g.send_and_recv(g.edges(), gcn_message, gcn_reduce)
        h = g.ndata.pop("h")
        # perform linear transformation
        return self.linear(h)

class GCN(nn.Module):

    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_feats, hidden_size)
        self.gcn2 = GCNLayer(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.gcn1(g, inputs)
        h = torch.relu(h)
        h = self.gcn2(g, h)
        return h

# visualization
def draw(i):
    cls1color = "#00FFFF"
    cls2color = "#FF00FF"
    pos = {}
    colors = []
    for v in range(nodes_num):
        index = list(nx_pos.keys())[v]
        pos[index] = all_logits[i][v].numpy()
        cls = pos[index].argmax()
        colors.append(cls1color if cls else cls2color)
    ax.cla()
    ax.set_title("Epoch: %d" % i)
    nx.draw_networkx(nx_G, nx_pos, node_color=colors, with_labels=True, node_size=300, ax=ax)

#train
if __name__ == '__main__':
    
    nx_G, nx_pos = get_network(country)
    G = dgl.from_networkx(nx_G)
    nodes_num, edge_num = G.number_of_nodes(), G.number_of_edges()
    net = GCN(nodes_num, hidden_size, 2)
    
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.set_tight_layout(False)
    nx.draw(nx_G, nx_pos, with_labels=True, node_color=[[0.5, 0.5, 0.5]])
    plt.show()
    print('%d nodes.' % nodes_num)
    print('%d edges.' % edge_num)
    print(net)

    inputs = torch.eye(nodes_num)
    labeled_nodes = torch.tensor([0, nodes_num-1])
    labels = torch.tensor([0, 1])

    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    all_logits = []

    for epoch in range(epoches):
        logits = net(G, inputs)
        all_logits.append(logits.detach())
        logp = F.log_softmax(logits, 1)

        # compute loss for labeled nodes
        loss = F.nll_loss(logp[labeled_nodes], labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Epoch %d | Loss: %.4f" % (epoch, loss.item()))

    fig = plt.figure(figsize=(16, 9))
    fig.clf()
    ax = fig.subplots()

    ani = animation.FuncAnimation(fig, draw, frames=len(all_logits), interval=200)
    plt.show()
