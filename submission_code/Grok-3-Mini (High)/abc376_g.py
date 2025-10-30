import sys
sys.setrecursionlimit(1000000)
MOD = 998244353

def dfs(node, children, a, counter, sum_ab):
    idx = counter[0]
    counter[0] += 1
    sum_ab[0] += a[node] * idx
    for child in children[node]:
        dfs(child, children, a, counter, sum_ab)

data = sys.stdin.read().split()
index = 0
T = int(data[index])
index += 1

for _ in range(T):
    N = int(data[index])
    index += 1
    p_values = list(map(int, data[index:index + N]))
    index += N
    a_values = list(map(int, data[index:index + N]))
    index += N
    
    # Create a_full with a[0] = 0
    a_full = [0] + a_values  # a_full[0] to a_full[N]
    
    # Compute S, sum of a_i from 1 to N
    S = sum(a_values)
    
    # Build children list
    children = [[] for _ in range(N + 1)]
    for i in range(N):
        node_idx = i + 1  # Node index from 1 to N
        parent = p_values[i]
        children[parent].append(node_idx)
    
    # Sort children by a descending for each node
    for i in range(N + 1):
        children[i].sort(key=lambda x: -a_full[x])
    
    # DFS to compute sum of a_v * index(v)
    counter = [1]  # Start index from 1 for node 0
    sum_ab_list = [0]
    dfs(0, children, a_full, counter, sum_ab_list)
    sum_ab_val = sum_ab_list[0]
    
    # Compute P and Q
    P = sum_ab_val - S
    P_mod = P % MOD
    if P_mod < 0:
        P_mod += MOD
    Q = S
    
    # Compute modular inverse of Q and then R
    inv_Q = pow(Q, MOD - 2, MOD)
    R = (P_mod * inv_Q) % MOD
    
    # Output R for this test case
    print(R)