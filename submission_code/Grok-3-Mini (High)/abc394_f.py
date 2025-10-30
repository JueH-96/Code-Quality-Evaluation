import sys
sys.setrecursionlimit(1000000)

data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
adj = [[] for _ in range(N + 1)]
for i in range(N - 1):
    a = int(data[index])
    b = int(data[index + 1])
    index += 2
    adj[a].append(b)
    adj[b].append(a)

max_size_ans = [-1]

def dfs(u, par):
    children = [v for v in adj[u] if v != par]
    child_dp_list = []
    for v in children:
        dp_v = dfs(v, u)
        child_dp_list.append(dp_v)
    num_children = len(children)
    
    # Compute A_u_p1
    if num_children >= 3:
        sizes_p1 = [dp[0] for dp in child_dp_list]  # A_c_p1_size
        sorted_sizes_p1 = sorted(sizes_p1, reverse=True)
        sum_top_three_p1 = sum(sorted_sizes_p1[:3])
        A_p1_size = 1 + sum_top_three_p1
        A_p1_has = True
    else:
        A_p1_size = 1
        A_p1_has = False
    
    # Compute A_u_p0
    options = []
    if num_children >= 1:
        # deg1 opt
        sizes_deg1 = [dp[0] for dp in child_dp_list]  # A_c_p1_size
        max_size_val_deg1 = max(sizes_deg1)
        for dp_c in child_dp_list:
            if dp_c[0] == max_size_val_deg1:
                has_deg4_deg1 = dp_c[1]  # A_c_p1_has_deg4
                break
        size_deg1_opt = 1 + max_size_val_deg1
        has_deg4_deg1_opt = has_deg4_deg1
        options.append((size_deg1_opt, has_deg4_deg1_opt))
    if num_children >= 4:
        # deg4 opt
        sizes_deg4_opt = [dp[0] for dp in child_dp_list]
        sorted_sizes_deg4 = sorted(sizes_deg4_opt, reverse=True)
        sum_top_four_deg4 = sum(sorted_sizes_deg4[:4])
        size_deg4_opt = 1 + sum_top_four_deg4
        has_deg4_deg4_opt = True
        options.append((size_deg4_opt, has_deg4_deg4_opt))
    if options:
        max_opt = max(options, key=lambda x: x[0])
        A_p0_size, A_p0_has = max_opt
    else:
        A_p0_size = -1
        A_p0_has = False
    
    # Update global max
    if A_p0_has and A_p0_size > max_size_ans[0]:
        max_size_ans[0] = A_p0_size
    
    # Return
    return (A_p1_size, A_p1_has, A_p0_size, A_p0_has)

# Call dfs from node 1
dfs(1, 0)

# Output the result
print(max_size_ans[0])