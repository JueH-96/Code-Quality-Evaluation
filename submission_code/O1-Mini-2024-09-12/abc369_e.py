# YOUR CODE HERE
import sys, sys
import sys
import sys, sys
import sys, sys
from heapq import heappush, heappop
import sys

def solve():
    import sys
    import sys
    from sys import stdin
    import sys
    def input():
        return sys.stdin.read()
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr]); ptr +=1
    M = int(data[ptr]); ptr +=1
    bridges = [None] * (M +1)
    adj = [[] for _ in range(N+1)]
    for b in range(1, M+1):
        u = int(data[ptr]); ptr +=1
        v = int(data[ptr]); ptr +=1
        t = int(data[ptr]); ptr +=1
        bridges[b] = (u, v, t)
        adj[u].append( (v, t, b) )
        adj[v].append( (u, t, b) )
    Q = int(data[ptr]); ptr +=1
    for _ in range(Q):
        K = int(data[ptr]); ptr +=1
        required = []
        for _ in range(K):
            b = int(data[ptr]); ptr +=1
            required.append(b)
        # Map bridge id to bit
        bridge_to_bit = {}
        for i, b in enumerate(required):
            bridge_to_bit[b] = i
        # Initialize Dijkstra
        from math import inf
        dist = [inf] * (N+1) * (1<<K)
        # To save memory, use a dict for distance
        # But faster is to use a list
        dist = [inf] * ((N+1) * (1<<K))
        start = 1
        mask_start = 0
        dist[start * (1<<K) + mask_start] = 0
        heap = []
        heappush(heap, (0, start, mask_start))
        visited = {}
        while heap:
            cost, u, mask = heappop(heap)
            idx = u * (1<<K) + mask
            if dist[idx] < cost:
                continue
            if u == N and mask == (1<<K)-1:
                print(cost)
                break
            for v, t, b in adj[u]:
                new_mask = mask
                if b in bridge_to_bit:
                    bit = bridge_to_bit[b]
                    new_mask = mask | (1 << bit)
                idx_new = v * (1<<K) + new_mask
                if dist[idx_new] > cost + t:
                    dist[idx_new] = cost + t
                    heappush(heap, (cost + t, v, new_mask))
        else:
            print(-1)