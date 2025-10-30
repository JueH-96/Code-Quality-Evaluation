# YOUR CODE HERE
import sys
from collections import defaultdict, deque

def solve():
    N, M, X1 = map(int, input().split())
    
    trains = []
    for i in range(M):
        a, b, s, t = map(int, input().split())
        trains.append((a, b, s, t))
    
    # Build graph of constraints
    # If train i arrives at city c at time t_i and train j departs from city c at time s_j
    # and t_i <= s_j, then we need X_j >= X_i + (t_i - s_j)
    
    # Group trains by arrival and departure cities
    arrivals = defaultdict(list)  # city -> list of (train_idx, arrival_time)
    departures = defaultdict(list)  # city -> list of (train_idx, departure_time)
    
    for i in range(M):
        a, b, s, t = trains[i]
        arrivals[b].append((i, t))
        departures[a].append((i, s))
    
    # Build constraint graph
    graph = defaultdict(list)  # train_i -> list of (train_j, weight)
    
    for city in range(1, N + 1):
        if city in arrivals and city in departures:
            for i, t_i in arrivals[city]:
                for j, s_j in departures[city]:
                    if t_i <= s_j:
                        # Add edge from i to j with weight (t_i - s_j)
                        graph[i].append((j, t_i - s_j))
    
    # Use Bellman-Ford to find minimum X values
    # X[0] = X1 is fixed
    X = [float('inf')] * M
    X[0] = X1
    
    # Relax edges M-1 times
    for _ in range(M - 1):
        updated = False
        for i in range(M):
            if X[i] != float('inf'):
                for j, weight in graph[i]:
                    if X[i] + weight < X[j]:
                        X[j] = X[i] + weight
                        updated = True
        if not updated:
            break
    
    # If any X value is still infinity, it means it's not reachable from train 1
    # In this case, we can set it to 0
    for i in range(M):
        if X[i] == float('inf'):
            X[i] = 0
    
    # Print X_2 through X_M
    print(' '.join(map(str, X[1:])))

solve()