class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        from functools import lru_cache
        
        n = len(nums)
        
        @lru_cache(None)
        def dp(idx, remaining_op1, remaining_op2):
            if idx == n:
                return 0
            
            current = nums[idx]
            min_sum = float('inf')
            
            # Option 1: No operation on current element
            min_sum = min(min_sum, current + dp(idx + 1, remaining_op1, remaining_op2))
            
            # Option 2: Only Operation 1 (divide by 2)
            if remaining_op1 > 0:
                new_val = (current + 1) // 2  # This is ceiling division
                min_sum = min(min_sum, new_val + dp(idx + 1, remaining_op1 - 1, remaining_op2))
            
            # Option 3: Only Operation 2 (subtract k)
            if remaining_op2 > 0 and current >= k:
                new_val = current - k
                min_sum = min(min_sum, new_val + dp(idx + 1, remaining_op1, remaining_op2 - 1))
            
            # Option 4: Both operations - try both orders
            if remaining_op1 > 0 and remaining_op2 > 0:
                # Order 1: Operation 1 first, then Operation 2
                after_op1 = (current + 1) // 2
                if after_op1 >= k:
                    after_both = after_op1 - k
                    min_sum = min(min_sum, after_both + dp(idx + 1, remaining_op1 - 1, remaining_op2 - 1))
                
                # Order 2: Operation 2 first, then Operation 1
                if current >= k:
                    after_op2 = current - k
                    after_both = (after_op2 + 1) // 2
                    min_sum = min(min_sum, after_both + dp(idx + 1, remaining_op1 - 1, remaining_op2 - 1))
            
            return min_sum
        
        return dp(0, op1, op2)