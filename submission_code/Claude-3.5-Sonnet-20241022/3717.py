class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        n = len(nums)
        
        # Helper function to calculate operations needed to make all elements equal to target
        def get_ops(arr, target):
            return sum(abs(num - target) for num in arr)
        
        # For each possible subarray of size x, calculate minimum operations
        def get_min_ops_for_window(start):
            window = nums[start:start + x]
            # The optimal target value will be the median of the window
            # For minimum operations, we should make all elements equal to one of the existing values
            min_ops = float('inf')
            for target in range(min(window), max(window) + 1):
                ops = get_ops(window, target)
                min_ops = min(min_ops, ops)
            return min_ops
        
        # For each starting position, calculate operations needed
        dp = []
        for i in range(n - x + 1):
            dp.append(get_min_ops_for_window(i))
        
        # Now we need to find k non-overlapping windows with minimum total operations
        # Dynamic programming to find minimum operations for selecting k windows
        @cache
        def solve(pos, remaining):
            if remaining == 0:
                return 0
            if pos > n - x:
                return float('inf')
            
            # Don't select current window
            result = solve(pos + 1, remaining)
            
            # Select current window
            if pos <= n - x:
                next_result = solve(pos + x, remaining - 1)
                if next_result != float('inf'):
                    result = min(result, dp[pos] + next_result)
            
            return result
        
        result = solve(0, k)
        return result if result != float('inf') else -1