def solve():
    N = int(input())
    s = input().strip()
    
    MOD = 998244353
    
    # Count the number of edges to vertex N
    k = s.count('1')
    
    # Total number of edges
    E = N + k
    
    # Degrees of each vertex
    deg = [2] * N + [k]
    for i in range(N):
        if s[i] == '1':
            deg[i] += 1
    
    # Use DP: dp[j] = number of ways to assign in-degrees to vertices processed so far
    # such that the sum of in-degrees is j
    dp = [0] * (E + 1)
    dp[0] = 1
    
    for i in range(N + 1):
        new_dp = [0] * (E + 1)
        for j in range(E + 1):
            if dp[j] == 0:
                continue
            # Vertex i can have in-degree from 0 to deg[i]
            for d in range(min(deg[i] + 1, E - j + 1)):
                new_dp[j + d] = (new_dp[j + d] + dp[j]) % MOD
        dp = new_dp
    
    print(dp[E])

solve()