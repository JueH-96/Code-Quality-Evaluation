def check_strongly_connected(n, adj):
    def dfs(start):
        visited = [False] * (2*n + 1)
        stack = [start]
        while stack:
            v = stack.pop()
            if not visited[v]:
                visited[v] = True
                for u in adj[v]:
                    if not visited[u]:
                        stack.append(u)
        return all(visited[1:])

    # Check if graph is strongly connected
    for i in range(1, 2*n + 1):
        if not dfs(i):
            return False
    return True

def solve(n, s):
    MOD = 998244353
    
    # Create adjacency list for initial edges
    adj = [[] for _ in range(2*n + 1)]
    for i in range(1, 2*n):
        adj[i].append(i+1)
    
    # Get white and black vertices
    white = []
    black = []
    for i in range(2*n):
        if s[i] == 'W':
            white.append(i+1)
        else:
            black.append(i+1)
            
    if len(white) != len(black):
        return 0
        
    def try_pairing(pairs):
        # Create copy of adjacency list
        curr_adj = [list(x) for x in adj]
        # Add edges for pairs
        for w, b in pairs:
            curr_adj[w].append(b)
        return check_strongly_connected(n, curr_adj)
    
    from itertools import permutations
    
    # Try all possible pairings of white and black vertices
    result = 0
    for black_perm in permutations(black):
        pairs = list(zip(white, black_perm))
        if try_pairing(pairs):
            result += 1
            
    return result % MOD

# Read input
n = int(input())
s = input().strip()

# Print output
print(solve(n, s))