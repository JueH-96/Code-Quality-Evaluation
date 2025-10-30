class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # Initialize dp arrays
        dp_include = [0] * n
        dp_exclude = [0] * n
        
        # Base cases
        dp_include[0] = max(0, nums[0])
        dp_exclude[0] = 0
        
        # Fill dp arrays for the initial nums
        for i in range(1, n):
            dp_include[i] = dp_exclude[i-1] + max(0, nums[i])
            dp_exclude[i] = max(dp_exclude[i-1], dp_include[i-1])
        
        result = 0
        
        # Process each query
        for pos, x in queries:
            old_value = nums[pos]
            nums[pos] = x
            
            # Update dp arrays considering the change at position pos
            if pos == 0:
                dp_include[0] = max(0, nums[0])
                dp_exclude[0] = 0
            else:
                dp_include[pos] = dp_exclude[pos-1] + max(0, nums[pos])
                dp_exclude[pos] = max(dp_exclude[pos-1], dp_include[pos-1])
            
            for i in range(pos + 1, n):
                dp_include[i] = dp_exclude[i-1] + max(0, nums[i])
                dp_exclude[i] = max(dp_exclude[i-1], dp_include[i-1])
            
            # The result for this query is the maximum of the last dp_include and dp_exclude
            result = (result + max(dp_include[-1], dp_exclude[-1])) % MOD
        
        return result