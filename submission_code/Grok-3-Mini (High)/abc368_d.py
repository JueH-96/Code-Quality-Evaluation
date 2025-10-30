import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
K = int(data[index])
index += 1
adj = [[] for _ in range(N + 1)]
for _ in range(N - 1):
    A = int(data[index])
    index += 1
    B = int(data[index])
    index += 1
    adj[A].append(B)
    adj[B].append(A)
is_S = [False] * (N + 1)
for _ in range(K):
    V = int(data[index])
    index += 1
    is_S[V] = True
est_count = [0]

def DFS(node, parent, adj, is_S, K, est_count):
    current_num = 1 if is_S[node] else 0
    for neighbor in adj[node]:
        if neighbor != parent:
            child_num = DFS(neighbor, node, adj, is_S, K, est_count)
            current_num += child_num
            if 1 <= child_num <= K - 1:
                est_count[0] += 1
    return current_num

DFS(1, 0, adj, is_S, K, est_count)
ans = est_count[0] + 1
print(ans)