import sys
from collections import deque

def main():
    import math
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        l, d, k, c, A, B = map(int, sys.stdin.readline().split())
        adj[A].append((l, d, k, c, B))
    
    INF = 10**18
    latest_departure = [-INF] * (N+1)
    latest_departure[N] = INF
    queue = deque()
    queue.append(N)
    
    while queue:
        S = queue.popleft()
        for seq in adj[S]:
            l_i, d_i, k_i, c_i, B = seq
            t_j_max = min(latest_departure[B] - c_i, l_i + (k_i - 1) * d_i)
            if t_j_max >= l_i:
                t_j = l_i + ((t_j_max - l_i) // d_i) * d_i
                arrival_time = t_j + c_i
                if arrival_time > latest_departure[S]:
                    latest_departure[S] = arrival_time
                    queue.append(S)
    
    for S in range(1, N):
        if latest_departure[S] < 0:
            print("Unreachable")
        else:
            print(latest_departure[S])

if __name__ == "__main__":
    main()