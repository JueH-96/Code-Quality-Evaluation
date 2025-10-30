def solve():
    import sys
    sys.setrecursionlimit(10**7)
    mod = 998244353
    
    S = sys.stdin.readline().strip()
    n = len(S)
    
    # dp[b] will represent the count of ways to form a valid partial parenthesis
    # with 'balance' = b after processing i characters
    dp = [0] * (n + 1)
    dp[0] = 1  # with 0 characters processed, balance 0 has exactly 1 way (empty)
    
    for i, ch in enumerate(S):
        new_dp = [0] * (n + 1)
        for bal in range(i + 1):  # possible balances range from 0..i
            if dp[bal] == 0:
                continue
            ways = dp[bal]
            if ch == '(':
                # If we put '(' then bal -> bal+1
                new_dp[bal + 1] = (new_dp[bal + 1] + ways) % mod
            elif ch == ')':
                # If we put ')' then bal -> bal-1 (if bal>0)
                if bal > 0:
                    new_dp[bal - 1] = (new_dp[bal - 1] + ways) % mod
            else:  # ch == '?'
                # Treat '?' as '('
                new_dp[bal + 1] = (new_dp[bal + 1] + ways) % mod
                # Treat '?' as ')'
                if bal > 0:
                    new_dp[bal - 1] = (new_dp[bal - 1] + ways) % mod
        dp = new_dp
    
    print(dp[0] % mod)