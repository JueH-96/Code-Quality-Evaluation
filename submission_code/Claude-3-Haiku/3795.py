class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        min_queries = float('inf')
        
        for i in range(len(queries)):
            l, r, val = queries[i]
            curr_sum = 0
            for j in range(l, r+1):
                curr_sum += max(0, nums[j] - val)
            if curr_sum == 0:
                min_queries = min(min_queries, i+1)
        
        return min_queries if min_queries != float('inf') else -1