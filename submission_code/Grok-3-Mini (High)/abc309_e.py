import sys
sys.setrecursionlimit(300010)
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1

# Build children list
children = [[] for _ in range(N + 1)]
for i in range(2, N + 1):
    p_i = int(data[index])
    index += 1
    children[p_i].append(i)

# Read insurances and compute max_y
max_y = [-1] * (N + 1)
for _ in range(M):
    x = int(data[index])
    index += 1
    y = int(data[index])
    index += 1
    if max_y[x] == -1:
        max_y[x] = y
    elif y > max_y[x]:
        max_y[x] = y

# Compute depth with BFS
depth = [-1] * (N + 1)
from collections import deque
queue = deque()
depth[1] = 0
queue.append(1)
while queue:
    node = queue.popleft()
    for child in children[node]:
        if depth[child] == -1:
            depth[child] = depth[node] + 1
            queue.append(child)

# Define DFS function
def dfs(u, par, max_cov_anc):
    if max_y[u] >= 0:  # has insurance
        cov_val_u = depth[u] + max_y[u]
        if max_cov_anc == -1:
            max_cov_incl = cov_val_u
        else:
            max_cov_incl = max(max_cov_anc, cov_val_u)
    else:  # no insurance
        if max_cov_anc == -1:
            max_cov_incl = -1
        else:
            max_cov_incl = max_cov_anc
    
    # Check if u is covered
    if max_cov_incl != -1 and max_cov_incl >= depth[u]:
        covered = 1
    else:
        covered = 0
    
    # Sum over children
    count = covered
    for v in children[u]:
        if v != par:
            child_count = dfs(v, u, max_cov_incl)
            count += child_count
    
    return count

# Call DFS from root and print answer
answer = dfs(1, 0, -1)
print(answer)