import sys
sys.setrecursionlimit(100010)

def dfs_compute_sub_and_dist(u, parent, dist, adj, weight, sub_weight_sum, S_sum_ref):
    S_sum_ref[0] += weight[u] * dist
    sub_weight_sum[u] = weight[u]
    for v in adj[u]:
        if v != parent:
            child_sub = dfs_compute_sub_and_dist(v, u, dist + 1, adj, weight, sub_weight_sum, S_sum_ref)
            sub_weight_sum[u] += child_sub
    return sub_weight_sum[u]

def dfs_reroot(u, parent, current_S, adj, sub_weight_sum, total_sum_weight, min_S_ref):
    min_S_ref[0] = min(min_S_ref[0], current_S)
    for v in adj[u]:
        if v != parent:
            S_v = current_S + total_sum_weight - 2 * sub_weight_sum[v]
            dfs_reroot(v, u, S_v, adj, sub_weight_sum, total_sum_weight, min_S_ref)

# main code
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
adj = [[] for _ in range(N + 1)]
for _ in range(N - 1):
    A = int(data[index])
    B = int(data[index + 1])
    index += 2
    adj[A].append(B)
    adj[B].append(A)
weight = [0] * (N + 1)
for i in range(1, N + 1):
    weight[i] = int(data[index])
    index += 1
total_sum_weight = sum(weight[1:])
sub_weight_sum = [0] * (N + 1)
S_sum_ref = [0]
dfs_compute_sub_and_dist(1, -1, 0, adj, weight, sub_weight_sum, S_sum_ref)
S_root = S_sum_ref[0]
min_S_ref = [float('inf')]
dfs_reroot(1, -1, S_root, adj, sub_weight_sum, total_sum_weight, min_S_ref)
print(min_S_ref[0])