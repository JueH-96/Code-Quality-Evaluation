import sys
import heapq

# Read input from standard input for efficiency.
N, A, B, C = map(int, sys.stdin.readline().split())
D = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]

def dijkstra(start_node, cost_func):
    """
    A standard Dijkstra's algorithm implementation for a complete graph.
    
    Args:
        start_node (int): The 0-indexed starting node.
        cost_func (function): A function to calculate the edge weight between two nodes.

    Returns:
        list[int]: A list of shortest distances from the start_node.
    """
    # Initialize distances to all nodes as infinity.
    distances = [float('inf')] * N
    distances[start_node] = 0
    
    # Priority queue storing tuples of (distance, node).
    pq = [(0, start_node)]
    
    while pq:
        dist, current_node = heapq.heappop(pq)
        
        # If a shorter path has already been found, skip.
        if dist > distances[current_node]:
            continue
        
        # Explore all other nodes as neighbors in the complete graph.
        for neighbor_node in range(N):
            if current_node == neighbor_node:
                continue
            
            edge_cost = cost_func(current_node, neighbor_node)
            new_dist = distances[current_node] + edge_cost
            
            if new_dist < distances[neighbor_node]:
                distances[neighbor_node] = new_dist
                heapq.heappush(pq, (new_dist, neighbor_node))
                
    return distances

# Define the cost function for car travel.
def car_cost(u, v):
    return D[u][v] * A

# Define the cost function for train travel.
def train_cost(u, v):
    return D[u][v] * B + C

# Step 1: Calculate shortest path costs from city 1 (index 0) by car.
# dist_car[k] stores the minimum time from city 1 to city k+1 by car.
dist_car = dijkstra(0, car_cost)

# Step 2: Calculate shortest path costs to city N (index N-1) by train.
# This is equivalent to running Dijkstra from N, as the graph is undirected.
# dist_train[k] stores the minimum time from city k+1 to city N by train.
dist_train = dijkstra(N - 1, train_cost)

# Step 3: Combine costs and find the overall minimum.
# Iterate through all possible cities 'k+1' to switch from car to train.
min_total_time = float('inf')
for k in range(N):
    # Total time = (time from 1 to k+1 by car) + (time from k+1 to N by train)
    total_time = dist_car[k] + dist_train[k]
    min_total_time = min(min_total_time, total_time)

# Print the final result.
print(min_total_time)