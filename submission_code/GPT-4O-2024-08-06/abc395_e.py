import sys
import heapq
from collections import defaultdict

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    X = int(data[2])
    
    forward_graph = defaultdict(list)
    reverse_graph = defaultdict(list)
    
    index = 3
    for _ in range(M):
        u = int(data[index])
        v = int(data[index + 1])
        forward_graph[u].append(v)
        reverse_graph[v].append(u)
        index += 2
    
    # Priority queue for Dijkstra-like BFS
    pq = []
    # (cost, vertex, reversed_state)
    heapq.heappush(pq, (0, 1, False))
    
    # Visited dictionary to store the minimum cost to reach each vertex in each state
    visited = {}
    
    while pq:
        current_cost, current_vertex, reversed_state = heapq.heappop(pq)
        
        # If we reached vertex N, return the cost
        if current_vertex == N:
            print(current_cost)
            return
        
        # If this state has been visited with a lower cost, skip it
        if (current_vertex, reversed_state) in visited and visited[(current_vertex, reversed_state)] <= current_cost:
            continue
        
        # Mark this state as visited with the current cost
        visited[(current_vertex, reversed_state)] = current_cost
        
        # Get the current graph based on the reversed state
        current_graph = reverse_graph if reversed_state else forward_graph
        
        # Move to adjacent vertices
        for neighbor in current_graph[current_vertex]:
            if (neighbor, reversed_state) not in visited or visited[(neighbor, reversed_state)] > current_cost + 1:
                heapq.heappush(pq, (current_cost + 1, neighbor, reversed_state))
        
        # Reverse the graph and stay at the current vertex
        new_reversed_state = not reversed_state
        if (current_vertex, new_reversed_state) not in visited or visited[(current_vertex, new_reversed_state)] > current_cost + X:
            heapq.heappush(pq, (current_cost + X, current_vertex, new_reversed_state))