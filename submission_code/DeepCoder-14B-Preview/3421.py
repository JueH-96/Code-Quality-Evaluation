class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Compute the prefix sum where each term is adjusted by (-1)^i
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i] * ((-1) ** i)
        
        max_plus_val = 0  # From j = -1
        max_minus_val = -float('inf')
        
        for i in range(n):
            x = prefix[i + 1]
            # Calculate the current maximum possible value for dp[i]
            current_max = max(max_plus_val + x, max_minus_val - x)
            
            # Update dp[i] (though we don't store it as we only need the final value)
            
            # Compute c2_i and c1_i for the current i
            c2_i = (-1) ** (i + 1)
            c1_i = current_max - c2_i * x  # current_max is dp[i]
            
            # Update the max_plus_val and max_minus_val based on c2_i
            if c2_i == 1:
                if c1_i > max_plus_val:
                    max_plus_val = c1_i
            else:
                if c1_i > max_minus_val:
                    max_minus_val = c1_i
        
        return current_max  # After the loop, current_max holds dp[n-1]