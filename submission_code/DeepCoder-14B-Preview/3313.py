class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Compute prefix sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        k_val = k
        INF = float('-inf')
        
        # Initialize DP table
        dp = [[INF] * n for _ in range(k_val + 1)]
        
        for j in range(k_val):
            c = (k_val - j) * ((-1) ** j)
            current_max = -INF
            if j == 0:
                current_max = 0  # Represents the state where no subarrays have been selected (l = -1)
            
            for i in range(n):
                if current_max != -INF:
                    new_val = current_max + c * prefix[i + 1]
                    if new_val > dp[j + 1][i]:
                        dp[j + 1][i] = new_val
                
                # Update current_max for the next i
                current = dp[j][i] - c * prefix[i + 1] if dp[j][i] != INF else -INF
                if current > current_max:
                    current_max = current
        
        # The answer is the maximum value in dp[k_val][i] for all i
        max_strength = max(dp[k_val])
        return max_strength if max_strength != INF else 0