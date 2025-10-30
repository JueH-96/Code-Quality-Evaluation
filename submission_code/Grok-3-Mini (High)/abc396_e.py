import sys
sys.setrecursionlimit(200005)
from collections import defaultdict

def find(node, parent, xor_to_parent):
    if parent[node] != node:
        p_root, xor_to_p_root = find(parent[node], parent, xor_to_parent)
        xor_node_to_p_root = xor_to_parent[node] ^ xor_to_p_root
        parent[node] = p_root
        xor_to_parent[node] = xor_node_to_p_root
        return p_root, xor_node_to_p_root
    else:
        return node, 0

data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1

X_list = [0] * M
Y_list = [0] * M
Z_list = [0] * M
for i in range(M):
    X_list[i] = int(data[index])
    Y_list[i] = int(data[index + 1])
    Z_list[i] = int(data[index + 2])
    index += 3

possible = True
ans = [0] * (N + 1)

for k in range(30):  # bits 0 to 29
    if not possible:
        break
    parent = [i for i in range(N + 1)]
    xor_to_parent = [0] * (N + 1)
    inconsistent = False
    for idx in range(M):
        u = X_list[idx]
        v = Y_list[idx]
        c = (Z_list[idx] >> k) & 1
        root_u, xor_u = find(u, parent, xor_to_parent)
        root_v, xor_v = find(v, parent, xor_to_parent)
        if root_u == root_v:
            if (xor_u ^ xor_v) != c:
                inconsistent = True
                possible = False
                break
        else:
            # union, set parent[root_v] = root_u
            xor_to_parent[root_v] = c ^ xor_u ^ xor_v
            parent[root_v] = root_u
    if inconsistent:
        continue  # skip to next k
    else:
        # consistent, now compress all nodes
        for i in range(1, N + 1):
            find(i, parent, xor_to_parent)
        # now compute size and sum_delta per component
        sum_delta_comp_dict = defaultdict(int)
        size_comp_dict = defaultdict(int)
        for i in range(1, N + 1):
            root = parent[i]
            delta = xor_to_parent[i]
            size_comp_dict[root] += 1
            sum_delta_comp_dict[root] += delta
        # now set a_ik for each i
        for i in range(1, N + 1):
            root_i = parent[i]
            sum_delta_root = sum_delta_comp_dict[root_i]
            size_root = size_comp_dict[root_i]
            if sum_delta_root <= size_root - sum_delta_root:
                a_ik = xor_to_parent[i]
            else:
                a_ik = 1 ^ xor_to_parent[i]
            # set the k-th bit in ans[i]
            ans[i] |= (a_ik << k)

if possible:
    print(' '.join(map(str, ans[1:])))
else:
    print(-1)