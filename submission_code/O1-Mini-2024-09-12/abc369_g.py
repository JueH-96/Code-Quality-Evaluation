# YOUR CODE HERE
import sys
import sys
import sys
from collections import deque

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    edges = [[] for _ in range(N+1)]
    total_sum = 0
    idx = 1
    for _ in range(N-1):
        U = int(N_and_rest[idx])
        V = int(N_and_rest[idx+1])
        L = int(N_and_rest[idx+2])
        edges[U].append((V, L))
        edges[V].append((U, L))
        total_sum += L
        idx +=3
    distances = [0]*(N+1)
    visited = [False]*(N+1)
    queue = deque()
    queue.append(1)
    visited[1]=True
    while queue:
        u = queue.popleft()
        for v, l in edges[u]:
            if not visited[v]:
                distances[v] = distances[u] + l
                visited[v]=True
                queue.append(v)
    distance_list = distances[1:]
    distance_list.sort(reverse=True)
    prefix_sum = [0]*(N+1)
    for i in range(1,N+1):
        prefix_sum[i] = prefix_sum[i-1] + distance_list[i-1]
    max_walk = 2 * total_sum
    for K in range(1,N+1):
        walk = 2 * prefix_sum[K]
        if walk > max_walk:
            walk = max_walk
        print(walk)

if __name__ == "__main__":
    main()