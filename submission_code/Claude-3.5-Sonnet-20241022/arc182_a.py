def solve():
    MOD = 998244353
    N, Q = map(int, input().split())
    P = []
    V = []
    for _ in range(Q):
        p, v = map(int, input().split())
        P.append(p)
        V.append(v)
    
    # dp[i][j] represents number of valid sequences after i operations
    # j=0: no restrictions, j=1: left restricted, j=2: right restricted
    dp = [[0]*3 for _ in range(Q+1)]
    dp[0][0] = 1
    
    for i in range(Q):
        p, v = P[i], V[i]
        
        # For each previous state
        for prev in range(3):
            if dp[i][prev] == 0:
                continue
                
            left_ok = True
            right_ok = True
            
            # Check if we can do left operation (1 to p)
            if prev == 0:
                left_ok = True
            elif prev == 1:
                left_ok = (v >= V[i-1])
            else:  # prev == 2
                if p >= P[i-1]:
                    left_ok = (v >= V[i-1])
                    
            # Check if we can do right operation (p to N)
            if prev == 0:
                right_ok = True
            elif prev == 1:
                if p <= P[i-1]:
                    right_ok = (v >= V[i-1])
            else:  # prev == 2
                right_ok = (v >= V[i-1])
                
            # Add valid transitions
            if left_ok:
                dp[i+1][1] = (dp[i+1][1] + dp[i][prev]) % MOD
            if right_ok:
                dp[i+1][2] = (dp[i+1][2] + dp[i][prev]) % MOD
    
    # Sum all possible final states
    ans = (dp[Q][1] + dp[Q][2]) % MOD
    print(ans)

solve()