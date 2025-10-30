# YOUR CODE HERE
import sys
import threading
from collections import defaultdict, deque

def main():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    e_to_sets = defaultdict(list)
    adj = [[] for _ in range(N)]
    S_contains_1 = []
    S_contains_M = []
    # To store whether each set contains 1 or M
    contains_1 = [False]*N
    contains_M = [False]*N
    for i in range(N):
        Ai = int(sys.stdin.readline())
        elements = list(map(int, sys.stdin.readline().split()))
        for e in elements:
            e_to_sets[e].append(i)
        if 1 in elements:
            contains_1[i] = True
        if M in elements:
            contains_M[i] = True
    # Build adjacency list
    for sets_list in e_to_sets.values():
        for idx in range(len(sets_list)-1):
            u = sets_list[idx]
            v = sets_list[idx+1]
            adj[u].append(v)
            adj[v].append(u)
    # BFS
    from collections import deque
    visited = [False]*N
    distance = [0]*N
    queue = deque()
    for i in range(N):
        if contains_1[i]:
            visited[i] = True
            distance[i] = 0
            queue.append(i)
    found = False
    while queue:
        u = queue.popleft()
        if contains_M[u]:
            print(distance[u])
            found = True
            break
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                distance[v] = distance[u] +1
                queue.append(v)
    if not found:
        print(-1)

threading.Thread(target=main).start()