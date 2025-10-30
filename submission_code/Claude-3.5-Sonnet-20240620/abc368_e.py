# YOUR CODE HERE
import sys
from collections import defaultdict

def read_input():
    N, M, X1 = map(int, input().split())
    trains = []
    for _ in range(M):
        A, B, S, T = map(int, input().split())
        trains.append((A, B, S, T))
    return N, M, X1, trains

def solve(N, M, X1, trains):
    graph = defaultdict(list)
    for i, (A, B, S, T) in enumerate(trains):
        graph[B].append((A, S, T, i + 1))

    delays = [0] * (M + 1)
    delays[1] = X1

    def dfs(city, delay):
        for prev_city, S, T, train_id in graph[city]:
            new_delay = max(0, delay - (S - T))
            if new_delay > delays[train_id]:
                delays[train_id] = new_delay
                dfs(prev_city, new_delay)

    dfs(trains[0][1], X1)

    return ' '.join(map(str, delays[2:]))

N, M, X1, trains = read_input()
result = solve(N, M, X1, trains)
print(result)