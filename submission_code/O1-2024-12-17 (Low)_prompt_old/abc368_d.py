def solve():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N, K = map(int, input_data[:2])
    edges_part = input_data[2:2 + 2*(N-1)]
    required_part = input_data[2 + 2*(N-1):]
    
    # Build adjacency list
    adj = [[] for _ in range(N+1)]
    idx = 0
    for _ in range(N-1):
        a = int(edges_part[idx]); b = int(edges_part[idx+1])
        idx += 2
        adj[a].append(b)
        adj[b].append(a)
    
    # Mark the required (K) vertices
    required = [False]*(N+1)
    for v in required_part:
        required[int(v)] = True
    
    # We can root the tree at any required vertex,
    # but if there's only one such vertex, just use that; otherwise use the first.
    root_candidates = [i for i in range(1, N+1) if required[i]]
    if not root_candidates:
        # No required vertices means the subtree is empty (but K>=1 in constraints, so this case shouldn't happen)
        print(0)
        return
    root = root_candidates[0]
    
    # sub_needed[u] = number of required vertices in the subtree rooted at u
    sub_needed = [0]*(N+1)
    
    # We'll do a DFS to fill in sub_needed
    sys.setrecursionlimit(10**7)
    
    def dfs(u, p):
        # Start with 1 if u is a required vertex, else 0
        count = 1 if required[u] else 0
        
        for nxt in adj[u]:
            if nxt == p:
                continue
            c = dfs(nxt, u)
            count += c
        sub_needed[u] = count
        return count
    
    dfs(root, -1)
    
    # The desired subtree is the set of u for which sub_needed[u] > 0
    # (meaning that subtree includes at least one required vertex).
    # The answer is just the count of nodes that have sub_needed[u] > 0.
    answer = sum(1 for u in range(1, N+1) if sub_needed[u] > 0)
    print(answer)

# If you want to test locally, you can uncomment the following lines and provide input:
# But remember, on the actual platform, this function will be called without arguments,
# and input will be taken from stdin, output to stdout.
# def test():
#     import io
#     input_data = """7 3
# 1 2
# 1 3
# 2 4
# 2 5
# 3 6
# 3 7
# 1 3 5
# """
#     print(run_io_fun(input_data, solve))

# solve()