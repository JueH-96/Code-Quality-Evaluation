class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        from functools import lru_cache
        
        n, m = len(nums), len(andValues)
        
        @lru_cache(None)
        def dp(i, j, current_and):
            # i: current index in nums
            # j: current index in andValues
            # current_and: current AND value of the ongoing subarray
            
            # Base case: processed all elements
            if i == n:
                return 0 if j == m else float('inf')
            
            # If we've used all andValues but still have elements left
            if j == m:
                return float('inf')
            
            # Update current AND value
            current_and &= nums[i]
            
            result = float('inf')
            
            # Option 1: Continue the current subarray
            if current_and >= andValues[j]:  # Only continue if AND can still match
                result = min(result, dp(i + 1, j, current_and))
            
            # Option 2: End current subarray here (if AND matches)
            if current_and == andValues[j]:
                # Start new subarray from next position
                next_result = dp(i + 1, j + 1, (1 << 17) - 1)  # Reset AND to all 1s
                if next_result != float('inf'):
                    result = min(result, nums[i] + next_result)
            
            return result
        
        # Start with all bits set (maximum possible AND value)
        result = dp(0, 0, (1 << 17) - 1)
        return result if result != float('inf') else -1