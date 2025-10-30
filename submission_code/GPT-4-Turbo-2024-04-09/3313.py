class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        from functools import lru_cache
        
        n = len(nums)
        
        # Prefix sum array for quick range sum calculation
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        # Helper function to get the sum of nums[l..r]
        def range_sum(l, r):
            return prefix_sum[r + 1] - prefix_sum[l]
        
        # Memoization for the dp function
        @lru_cache(None)
        def dp(i, remaining_k):
            if remaining_k == 0:
                return 0
            if i >= n:
                return float('-inf')  # Not enough elements left to form remaining_k subarrays
            
            # Option 1: Do not start a subarray at index i
            best = dp(i + 1, remaining_k)
            
            # Option 2: Start a subarray at index i and consider all possible ends of this subarray
            current_sum = 0
            for j in range(i, n):
                current_sum += nums[j]
                if remaining_k > 1:
                    # Calculate the strength and continue to the next subarrays
                    best = max(best, current_sum * remaining_k + dp(j + 1, remaining_k - 1))
                else:
                    # If it's the last subarray, we just add its sum
                    best = max(best, current_sum)
            
            return best
        
        # Start the dp from index 0 with k subarrays to form
        return dp(0, k)