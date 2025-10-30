from collections import deque
import itertools
import sys

data = sys.stdin.read().split()
index = 0
N = int(data[index])
M = int(data[index+1])
K = int(data[index+2])
index += 3

edges = []
 for i in range(M):
    u = int(data[index])
    v = int(data[index+1])
    edges.append((u, v))
    index += 2

# Create adjacency list with edge indices
adj = [[] for _ in range(N+1)]
 for i in range(M):
    u, v = edges[i]
    adj[u].append((v, i))  # v and edge index

# Binary search for the maximum D
left = 0
 right = N-1
 ans = 0
 while left <= right:
    mid = (left + right) // 2
    if check(mid, K, edges, adj, M, N):
        ans = mid
        left = mid + 1
    else:
        right = mid - 1

 print(ans)

 def check(D, K, edges, adj, M, N):
    for comb in itertools.combinations(range(M), K):
        S = set( comb )  # set of edge indices in S
        dist_n = min_dist(S, adj, N)
        if dist_n == -1 or dist_n >= D:
            return True
    return False

 def min_dist(S, adj, N):
    dist = [-1] * (N+1)
    dist[1] = 0
    q = deque()
    q.append((1, 0))  # node, dist_sum

    while q:
        cur_node, cur_dist = q.popleft()
        for v, edge_idx in adj[cur_node]:
            weight = 1 if edge_idx in S else 0
            new_dist = cur_dist + weight
            if dist[v] == -1 or new_dist < dist[v]:
                dist[v] = new_dist
                if weight == 0:
                    q.appendleft((v, new_dist))  # add to front for weight 0
                else:
                    q.append((v, new_dist))  # add to back for weight 1

    return dist[N]