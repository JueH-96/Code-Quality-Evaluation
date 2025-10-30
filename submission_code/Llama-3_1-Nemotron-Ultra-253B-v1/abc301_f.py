MOD = 998244353

S = input().strip()

# Initialize DP: state 0, 1_X (26), 2_X (26), 3_X (26), 4
dp = [0] * 80
dp[0] = 1

for c in S:
    new_dp = [0] * 80
    upper = 0
    lower = 0
    if c == '?':
        upper = 26
        lower = 26
    else:
        if c.isupper():
            upper = 1
        else:
            lower = 1
    
    # Process state 0
    if upper > 0:
        new_dp[0] = (new_dp[0] + dp[0] * upper) % MOD
        if c == '?':
            for x in range(26):
                new_dp[1 + x] = (new_dp[1 + x] + dp[0]) % MOD
        else:
            x = ord(c) - ord('A')
            new_dp[1 + x] = (new_dp[1 + x] + dp[0]) % MOD
    if lower > 0:
        new_dp[0] = (new_dp[0] + dp[0] * lower) % MOD
    
    # Process state 1_X (each X)
    for x in range(26):
        state = 1 + x
        if dp[state] == 0:
            continue
        val = dp[state]
        # Uppercase transitions
        if upper > 0:
            if c == '?':
                # Y can be any uppercase
                # Y == X
                new_dp[27 + x] = (new_dp[27 + x] + val) % MOD
                new_dp[state] = (new_dp[state] + val) % MOD
                # Y != X
                for y in range(26):
                    if y != x:
                        new_dp[1 + y] = (new_dp[1 + y] + val) % MOD
                        new_dp[state] = (new_dp[state] + val) % MOD
            else:
                y = ord(c) - ord('A')
                if y == x:
                    new_dp[27 + x] = (new_dp[27 + x] + val) % MOD
                    new_dp[state] = (new_dp[state] + val) % MOD
                else:
                    new_dp[1 + y] = (new_dp[1 + y] + val) % MOD
                    new_dp[state] = (new_dp[state] + val) % MOD
        # Lowercase transitions
        if lower > 0:
            new_dp[state] = (new_dp[state] + val * lower) % MOD
    
    # Process state 2_X (each X)
    for x in range(26):
        state = 27 + x
        if dp[state] == 0:
            continue
        val = dp[state]
        # Uppercase transitions
        if upper > 0:
            if c == '?':
                for y in range(26):
                    new_dp[1 + y] = (new_dp[1 + y] + val) % MOD
                    new_dp[state] = (new_dp[state] + val) % MOD
            else:
                y = ord(c) - ord('A')
                new_dp[1 + y] = (new_dp[1 + y] + val) % MOD
                new_dp[state] = (new_dp[state] + val) % MOD
        # Lowercase transitions
        if lower > 0:
            new_dp[53 + x] = (new_dp[53 + x] + val * lower) % MOD
            new_dp[state] = (new_dp[state] + val * lower) % MOD
    
    # Process state 3_X (each X)
    for x in range(26):
        state = 53 + x
        if dp[state] == 0:
            continue
        val = dp[state]
        # Uppercase transitions
        if upper > 0:
            new_dp[79] = (new_dp[79] + val * upper) % MOD
            new_dp[state] = (new_dp[state] + val * upper) % MOD
        # Lowercase transitions
        if lower > 0:
            new_dp[state] = (new_dp[state] + val * lower) % MOD
    
    # Process state 4
    new_dp[79] = (new_dp[79] + dp[79] * (upper + lower)) % MOD
    
    dp = new_dp

answer = sum(dp[:79]) % MOD
print(answer)