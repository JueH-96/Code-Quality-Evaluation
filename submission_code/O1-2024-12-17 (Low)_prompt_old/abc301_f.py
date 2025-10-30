def solve():
    import sys
    sys.setrecursionlimit(10**7)
    MOD = 998244353

    S = sys.stdin.readline().rstrip()
    n = len(S)

    # dp0:  number of ways (so far) that have NOT matched any part of the "DDoS" subsequence
    # dp1[x]: number of ways that have matched exactly the first uppercase letter X (and are ready for the second X)
    # dp2[x]: number of ways that have matched uppercase letters X, X (and are ready for a lowercase letter)
    # dp3[x]: number of ways that have matched X, X, lowercase (and are ready for the final uppercase)
    # dp4:  number of ways that have already formed a "DDoS"-type subsequence
    
    # We only keep track of not-yet-completed patterns in dp0, dp1, dp2, dp3.
    # Once dp4 is reached, it means at least one "DDoS"-type subsequence can be formed; dp4 is absorbing.

    dp0 = 1
    dp1 = [0]*26
    dp2 = [0]*26
    dp3 = [0]*26
    dp4 = 0

    for c in S:
        if c == '?':
            # c can be any uppercase (26 ways) or any lowercase (26 ways)
            up_count = 26
            low_count = 26
            # For each uppercase letter X, exactly 1 way among the up_count if we choose X
            up_spec = [1]*26
        elif 'A' <= c <= 'Z':
            up_count = 1
            low_count = 0
            x = ord(c) - ord('A')
            up_spec = [0]*26
            up_spec[x] = 1
        else:
            # c is a lowercase letter
            up_count = 0
            low_count = 1
            up_spec = [0]*26

        # Prepare new DP arrays
        new0 = 0
        new1 = [0]*26
        new2 = [0]*26
        new3 = [0]*26
        new4 = 0

        all_ways = (up_count + low_count) % MOD
        
        # 1) Skipping the character in each state:
        # dp0 -> dp0
        new0 = (new0 + dp0 * all_ways) % MOD
        
        # dp4 -> dp4 (once matched, remain matched)
        new4 = (new4 + dp4 * all_ways) % MOD
        
        # dp1[x] -> dp1[x]
        for x in range(26):
            new1[x] = (new1[x] + dp1[x]*all_ways) % MOD
        
        # dp2[x] -> dp2[x]
        for x in range(26):
            new2[x] = (new2[x] + dp2[x]*all_ways) % MOD
        
        # dp3[x] -> dp3[x]
        for x in range(26):
            new3[x] = (new3[x] + dp3[x]*all_ways) % MOD
        
        # 2) Using the character to advance the subsequence:
        # dp0 -> dp1[x] if character is uppercase X
        for x in range(26):
            if up_spec[x] != 0:  # up_spec[x] is either 0 or 1
                val = dp0 * up_spec[x]
                new1[x] = (new1[x] + val) % MOD
        
        # dp1[x] -> dp2[x] if character is uppercase X
        for x in range(26):
            if up_spec[x] != 0:
                val = dp1[x] * up_spec[x]
                new2[x] = (new2[x] + val) % MOD
        
        # dp2[x] -> dp3[x] if character is lowercase
        if low_count != 0:
            for x in range(26):
                if dp2[x] != 0:
                    val = dp2[x] * low_count
                    new3[x] = (new3[x] + val) % MOD
        
        # dp3[x] -> dp4 if character is uppercase
        sum_dp3 = sum(dp3) % MOD
        if up_count != 0:
            # any uppercase letter will complete the pattern from dp3[x]
            val = sum_dp3 * up_count
            new4 = (new4 + val) % MOD
        
        # Update dp arrays
        dp0 = new0
        dp1 = new1
        dp2 = new2
        dp3 = new3
        dp4 = new4

    # The number of ways that do NOT contain a "DDoS"-type subsequence
    # is dp0 + sum of dp1 + sum of dp2 + sum of dp3 after processing all characters.
    ans = dp0
    ans = (ans + sum(dp1)) % MOD
    ans = (ans + sum(dp2)) % MOD
    ans = (ans + sum(dp3)) % MOD

    print(ans % MOD)