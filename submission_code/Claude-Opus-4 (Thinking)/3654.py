class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        n = len(nums)
        memo = {}
        
        def dp(idx, rem_op1, rem_op2):
            if idx == n:
                return 0
            
            if (idx, rem_op1, rem_op2) in memo:
                return memo[(idx, rem_op1, rem_op2)]
            
            x = nums[idx]
            
            # Option 1: No operation
            result = x + dp(idx + 1, rem_op1, rem_op2)
            
            # Option 2: Only op1
            if rem_op1 > 0:
                after_op1 = (x + 1) // 2
                result = min(result, after_op1 + dp(idx + 1, rem_op1 - 1, rem_op2))
            
            # Option 3: Only op2
            if rem_op2 > 0 and x >= k:
                after_op2 = x - k
                result = min(result, after_op2 + dp(idx + 1, rem_op1, rem_op2 - 1))
            
            # Option 4: Both operations
            if rem_op1 > 0 and rem_op2 > 0:
                best_both = float('inf')
                
                # Order 1: op1 first, then op2
                after_op1 = (x + 1) // 2
                if after_op1 >= k:
                    after_both = after_op1 - k
                    best_both = min(best_both, after_both)
                
                # Order 2: op2 first, then op1
                if x >= k:
                    after_op2 = x - k
                    after_both = (after_op2 + 1) // 2
                    best_both = min(best_both, after_both)
                
                if best_both != float('inf'):
                    result = min(result, best_both + dp(idx + 1, rem_op1 - 1, rem_op2 - 1))
            
            memo[(idx, rem_op1, rem_op2)] = result
            return result
        
        return dp(0, op1, op2)