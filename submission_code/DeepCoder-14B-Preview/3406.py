MOD = 10**9 + 7

class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        n = zero + one
        if n == 0:
            return 0
        
        # Initialize DP as a list of dictionaries
        dp = [{} for _ in range(n)]
        
        # Base case: i = 0 (first element)
        if zero >= 1:
            key = (0, 1, 1, 0)
            dp[0][key] = dp[0].get(key, 0) + 1
        if one >= 1:
            key = (1, 1, 0, 1)
            dp[0][key] = dp[0].get(key, 0) + 1
        
        for i in range(n - 1):
            current = dp[i]
            next_dp = {}
            for (last, run, z, o), count in current.items():
                # Try adding the same element
                if last == 0:
                    new_run = run + 1
                    if new_run <= limit:
                        new_z = z + 1
                        new_o = o
                        if new_z <= zero:
                            key = (0, new_run, new_z, new_o)
                            if key in next_dp:
                                next_dp[key] = (next_dp[key] + count) % MOD
                            else:
                                next_dp[key] = count % MOD
                    # Try adding 1
                    new_run = 1
                    new_z = z
                    new_o = o + 1
                    if new_o <= one:
                        key = (1, new_run, new_z, new_o)
                        if key in next_dp:
                            next_dp[key] = (next_dp[key] + count) % MOD
                        else:
                            next_dp[key] = count % MOD
                else:  # last == 1
                    new_run = run + 1
                    if new_run <= limit:
                        new_o = o + 1
                        if new_o <= one:
                            key = (1, new_run, z, new_o)
                            if key in next_dp:
                                next_dp[key] = (next_dp[key] + count) % MOD
                            else:
                                next_dp[key] = count % MOD
                    # Try adding 0
                    new_run = 1
                    new_z = z + 1
                    if new_z <= zero:
                        key = (0, new_run, new_z, o)
                        if key in next_dp:
                            next_dp[key] = (next_dp[key] + count) % MOD
                        else:
                            next_dp[key] = count % MOD
            dp[i + 1] = next_dp
        
        # Calculate the result by summing valid states in the last DP layer
        result = 0
        for (last, run, z, o), count in dp[n - 1].items():
            if z == zero and o == one:
                result = (result + count) % MOD
        
        return result