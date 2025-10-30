def solve():
    import sys
    input = sys.stdin.read
    MOD = 998244353
    
    S = input().strip()
    n = len(S)
    
    # Edge case: if length of S is less than 4, no DDoS-type string can be formed
    if n < 4:
        print(pow(52, S.count('?'), MOD))
        return
    
    # dp[i][j] will represent the number of ways to fill the string up to index i
    # such that the last j characters can potentially start forming a DDoS-type string
    # j = 0: general count of valid strings up to i
    # j = 1: count of strings ending with an uppercase letter (potential start of DDoS)
    # j = 2: count of strings ending with two same uppercase letters
    # j = 3: count of strings ending with two same uppercase letters followed by a lowercase
    # j = 4: count of strings ending with a DDoS-type string (invalid state)
    
    dp = [[0] * 5 for _ in range(n + 1)]
    dp[0][0] = 1  # Empty string is a valid string with no DDoS
    
    for i in range(n):
        char = S[i]
        if char == '?':
            # '?' can be any of 26 lowercase or 26 uppercase letters
            # Update dp[i+1][j] from dp[i][j]
            for j in range(5):
                dp[i+1][j] = dp[i][j] * 52 % MOD
            
            # Transition for forming DDoS-type strings
            # '?' as an uppercase letter
            for j in range(1, 5):
                dp[i+1][j] += dp[i][j-1] * 26 % MOD
                dp[i+1][j] %= MOD
            
            # '?' as a lowercase letter
            dp[i+1][3] += dp[i][2] * 26 % MOD
            dp[i+1][3] %= MOD
        else:
            # Normal character, direct transitions
            if 'A' <= char <= 'Z':
                # Uppercase transitions
                dp[i+1][1] = (dp[i+1][1] + dp[i][0]) % MOD
                if i > 0 and S[i-1] == char:
                    dp[i+1][2] = (dp[i+1][2] + dp[i][1]) % MOD
                dp[i+1][4] = (dp[i+1][4] + dp[i][3]) % MOD
            if 'a' <= char <= 'z':
                # Lowercase transitions
                dp[i+1][3] = (dp[i+1][3] + dp[i][2]) % MOD
            
            # Copy over the unaffected dp values
            for j in [0, 1, 2, 3, 4]:
                if j != 1 and j != 2 and j != 3 and j != 4:
                    dp[i+1][j] = (dp[i+1][j] + dp[i][j]) % MOD
    
    # The answer is the total valid strings of length n that do not end with a DDoS-type string
    result = (dp[n][0] + dp[n][1] + dp[n][2] + dp[n][3]) % MOD
    print(result)