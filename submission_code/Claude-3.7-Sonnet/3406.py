class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        
        # Initialize DP table
        # dp[z][o][last][consec] represents number of ways to form an array with:
        # - z zeros, o ones
        # - last digit is either 0 (last=0) or 1 (last=1)
        # - consec consecutive occurrences of the last digit at the end
        dp = [[[[0 for _ in range(limit + 1)] for _ in range(2)] for _ in range(one + 1)] for _ in range(zero + 1)]
        
        # Base cases: single element arrays
        if zero > 0:
            dp[1][0][0][1] = 1  # Array with a single 0
        if one > 0:
            dp[0][1][1][1] = 1  # Array with a single 1
        
        for z in range(zero + 1):
            for o in range(one + 1):
                if z + o <= 1:  # Skip the base cases
                    continue
                
                # Transitions for last element being 0
                if z > 0:
                    # Case 1: Adding a 0 after a sequence of 1s
                    for prev_consec in range(1, min(o + 1, limit + 1)):
                        dp[z][o][0][1] = (dp[z][o][0][1] + dp[z-1][o][1][prev_consec]) % MOD
                    
                    # Case 2: Adding a 0 after a sequence of 0s
                    for consec in range(2, min(limit + 1, z + 1)):
                        dp[z][o][0][consec] = (dp[z][o][0][consec] + dp[z-1][o][0][consec-1]) % MOD
                
                # Transitions for last element being 1
                if o > 0:
                    # Case 3: Adding a 1 after a sequence of 0s
                    for prev_consec in range(1, min(z + 1, limit + 1)):
                        dp[z][o][1][1] = (dp[z][o][1][1] + dp[z][o-1][0][prev_consec]) % MOD
                    
                    # Case 4: Adding a 1 after a sequence of 1s
                    for consec in range(2, min(limit + 1, o + 1)):
                        dp[z][o][1][consec] = (dp[z][o][1][consec] + dp[z][o-1][1][consec-1]) % MOD
        
        # Calculate the final answer
        result = 0
        for last in range(2):
            for consec in range(1, limit + 1):
                if (last == 0 and consec <= zero) or (last == 1 and consec <= one):
                    result = (result + dp[zero][one][last][consec]) % MOD
        
        return result