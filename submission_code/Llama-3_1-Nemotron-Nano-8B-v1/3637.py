MOD = 10**9 + 7

class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        n = len(num)
        digits = list(map(int, num))
        total_sum = sum(digits)
        
        # If the total sum is odd, no balanced permutations possible
        if total_sum % 2 != 0:
            return 0
        
        target = total_sum // 2
        m = (n + 1) // 2  # Number of even indices
        
        # Count the frequency of each digit
        cnt = [0] * 10
        for d in digits:
            cnt[d] += 1
        
        # Precompute factorial and inverse factorial up to the maximum needed value (80)
        max_fact = 80
        fact = [1] * (max_fact + 1)
        for i in range(1, max_fact + 1):
            fact[i] = fact[i-1] * i % MOD
        
        inv_fact = [1] * (max_fact + 1)
        inv_fact[max_fact] = pow(fact[max_fact], MOD - 2, MOD)
        for i in range(max_fact - 1, -1, -1):
            inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD
        
        # Initialize DP table where dp[y][z] represents the number of ways to get y digits and sum z
        dp = [[0] * (target + 1) for _ in range(m + 1)]
        dp[0][0] = 1  # Base case: 0 digits used, sum 0
        
        for d in range(10):
            if cnt[d] == 0:
                continue  # Skip digits not present in the input
            
            # Create a new DP table for the current digit
            new_dp = [[0] * (target + 1) for _ in range(m + 1)]
            
            for y in range(m + 1):
                for z in range(target + 1):
                    if dp[y][z] == 0:
                        continue
                    
                    # Try all possible counts k of the current digit in even positions
                    max_k = min(cnt[d], m - y, (target - z) // d) if d != 0 else min(cnt[d], m - y)
                    for k in range(0, max_k + 1):
                        new_y = y + k
                        new_z = z + d * k
                        if new_y > m or new_z > target:
                            continue
                        
                        # Calculate the term: 1/(k! * (cnt[d] - k)!)
                        term = inv_fact[k] * inv_fact[cnt[d] - k] % MOD
                        new_dp[new_y][new_z] = (new_dp[new_y][new_z] + dp[y][z] * term) % MOD
            
            # Update dp to new_dp for the next digit
            dp = new_dp
        
        # Calculate the final answer
        ans = dp[m][target] * fact[m] % MOD
        ans = ans * fact[n - m] % MOD
        return ans