class Solution:
    def maxSum(self, nums: List[int], k: int, m: int) -> int:
        n = len(nums)
        
        # Memoization cache
        memo = {}
        
        def dp(pos, remaining_k):
            # Base cases
            if remaining_k == 0:
                return 0
            if pos >= n:
                return float('-inf')  # Invalid state
            
            # Check if not enough elements left to form remaining_k subarrays of length m
            if n - pos < remaining_k * m:
                return float('-inf')
            
            if (pos, remaining_k) in memo:
                return memo[(pos, remaining_k)]
            
            # Option 1: Skip current position
            result = dp(pos + 1, remaining_k)
            
            # Option 2: Take a subarray starting at pos
            current_sum = 0
            for length in range(m, n - pos + 1):
                current_sum += nums[pos + length - 1]
                # Take subarray of this length and solve for remaining
                next_result = dp(pos + length, remaining_k - 1)
                if next_result != float('-inf'):
                    result = max(result, current_sum + next_result)
            
            memo[(pos, remaining_k)] = result
            return result
        
        return dp(0, k)