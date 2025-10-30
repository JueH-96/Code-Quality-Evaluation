# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    import math
    import heapq
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline
    N1, N2, M = map(int, sys.stdin.readline().split())
    N = N1 + N2
    adjA = [[] for _ in range(N1+1)]  # Nodes 1 to N1
    adjB = [[] for _ in range(N2+1)]  # Nodes N1+1 to N1+N2
    for _ in range(M):
        a, b = map(int, sys.stdin.readline().split())
        if a <= N1 and b <= N1:
            # Both in component A
            adjA[a].append(b)
            adjA[b].append(a)
        elif a > N1 and b > N1:
            # Both in component B
            a_idx = a - N1
            b_idx = b - N1
            adjB[a_idx].append(b_idx)
            adjB[b_idx].append(a_idx)
        else:
            # Ignore edges between components
            pass

    from collections import deque

    # BFS in component A from node 1
    distA = [-1] * (N1 +1)
    queue = deque()
    queue.append(1)
    distA[1] = 0
    while queue:
        u = queue.popleft()
        for v in adjA[u]:
            if distA[v] == -1:
                distA[v] = distA[u] +1
                queue.append(v)
    max_distance_A = max(distA)

    # BFS in component B from node N1+N2
    startB = N2  # Index of node N1+N2 in adjB (since N1+N2 - N1 = N2)
    distB = [-1] * (N2 +1)
    queue = deque()
    queue.append(startB)
    distB[startB] = 0
    while queue:
        u = queue.popleft()
        for v in adjB[u]:
            if distB[v] == -1:
                distB[v] = distB[u] +1
                queue.append(v)
    max_distance_B = max(distB)

    # Answer is max_distance_A + 1 + max_distance_B
    answer = max_distance_A + 1 + max_distance_B
    print(answer)

threading.Thread(target=main).start()