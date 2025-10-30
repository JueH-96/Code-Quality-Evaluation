import sys
from collections import defaultdict 

class Graph:
    def __init__(self):
        self.vertices = {}
        self.layers = {}
        self.dept = {}
        self.layer_cnt = 0
    
    def add_edge(self,u,v):
        if u not in self.vertices:
            self.vertices[u]= []
        if v not in self.vertices:
            self.vertices[v]= []

        self.vertices[u].append(v)
        self.vertices[v].append(u)

    def dfs(self, vertex, parent, depth):
        self.dept[vertex]= depth
        for nxt in self.vertices[vertex]:
            if nxt != parent:
                self.dfs(nxt, vertex, depth+1)
    
    def add_layer(self, vertex, parent):
        if len(self.vertices[vertex]) == 1 and vertex != 1:
            # add vertex to layer
            if self.layer_cnt not in self.layers:
                self.layers[self.layer_cnt]= []
            self.layers[self.layer_cnt].append(vertex)
        
        for nxt in self.vertices[vertex]:
            if nxt != parent:
                self.add_layer(nxt, vertex)

        return self.layers

            
    def destroy_cnt(self):
        self.dfs(1,None,0)
        self.add_layer(1,None)

        destroy_layer_cnt = defaultdict(int)
        
        for layer in range(self.layer_cnt,-1,-1):
            # dfs from root(1) and follow layer
            for v in self.layers[layer]:
                if len(self.vertices[v]) == 1 and 1 not in self.vertices[v]:
                    # destroy layer
                    if layer in destroy_layer_cnt:
                        destroy_layer_cnt[layer] += 1
                    else:
                        destroy_layer_cnt[layer] = 1
                    self.vertices[v]= []  #like destroying leaf node
                
                else:
                    # destroy layer
                    if layer in destroy_layer_cnt:  # root is not included
                        destroy_layer_cnt[layer] += 1
                    else:
                        destroy_layer_cnt[layer] = 1
                
                    # destroy root that connected to the vertex on upper layer
                    if self.dept[1] - self.dept[v] == 1 and  1 in self.vertices[v]:
                        destroy_layer_cnt[layer-1] += 1
                        self.vertices[1]= []
                    
                    for nxt in self.vertices[v]:
                        destroy_layer_cnt[self.dept[nxt]-1] += 1
        
        layers = sorted(destroy_layer_cnt.keys(), reverse=True)
        max_layered_destroy = 0
        for k in layers:
            if k >= max_layered_destroy: 
                max_layered_destroy = k
        
        return destroy_layer_cnt[max_layered_destroy]+ len(destroy_layer_cnt)-1

if __name__ == "__main__":
    n = int(sys.stdin.readline())
    g = Graph()
    
    for _ in range(n-1):
        u,v = map(int, sys.stdin.readline().split())
        g.add_edge(u, v)
    print(g.destroy_cnt())