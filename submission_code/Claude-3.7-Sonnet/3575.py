class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        memo = {}
        
        def dfs(idx, count, left_or, right_or):
            # Base case: we've selected exactly 2*k elements
            if count == 2 * k:
                return left_or ^ right_or
            
            # If we've reached the end of the array or can't form a valid subsequence
            if idx == n or count + (n - idx) < 2 * k:
                return float('-inf')
            
            # Check if we've already computed this state
            if (idx, count, left_or, right_or) in memo:
                return memo[(idx, count, left_or, right_or)]
            
            # Skip the current element
            result = dfs(idx + 1, count, left_or, right_or)
            
            # Include the current element
            if count < k:  # Include in left part
                result = max(result, dfs(idx + 1, count + 1, left_or | nums[idx], right_or))
            elif count < 2 * k:  # Include in right part
                result = max(result, dfs(idx + 1, count + 1, left_or, right_or | nums[idx]))
            
            # Memoize and return
            memo[(idx, count, left_or, right_or)] = result
            return result
        
        return dfs(0, 0, 0, 0)