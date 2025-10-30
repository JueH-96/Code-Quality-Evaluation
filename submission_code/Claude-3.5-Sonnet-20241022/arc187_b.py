def find_components(adj, n):
    visited = [False] * n
    components = 0
    
    def dfs(v):
        visited[v] = True
        for u in range(n):
            if adj[v][u] and not visited[u]:
                dfs(u)
    
    for v in range(n):
        if not visited[v]:
            dfs(v)
            components += 1
    
    return components

def solve_sequence(sequence, n):
    # Create adjacency matrix
    adj = [[False] * n for _ in range(n)]
    
    # Add edges where A_i <= A_j
    for i in range(n):
        for j in range(i+1, n):
            if sequence[i] <= sequence[j]:
                adj[i][j] = adj[j][i] = True
    
    return find_components(adj)

def main():
    MOD = 998244353
    
    # Read input
    N, M = map(int, input().split())
    B = list(map(int, input().split()))
    
    # Find positions of -1
    minus_one_pos = [i for i, x in enumerate(B) if x == -1]
    q = len(minus_one_pos)
    
    # If no -1s, just solve the sequence directly
    if q == 0:
        print(solve_sequence(B, N))
        return
    
    # Try all possible combinations
    result = 0
    
    # For each possible combination
    for mask in range(M ** q):
        # Create new sequence
        curr_sequence = B.copy()
        temp = mask
        
        # Replace each -1 with a value from 1 to M
        for pos in minus_one_pos:
            curr_sequence[pos] = (temp % M) + 1
            temp //= M
        
        # Add f(B') to result
        result = (result + solve_sequence(curr_sequence, N)) % MOD
    
    print(result)

if __name__ == "__main__":
    main()