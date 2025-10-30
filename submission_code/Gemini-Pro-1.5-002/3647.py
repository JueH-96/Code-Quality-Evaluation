class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        m = len(queries)
        diffs = [0] * (n + 1)
        for l, r in queries:
            diffs[l] += 1
            diffs[r + 1] -= 1
        
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i+1] = prefix_sum[i] + diffs[i]
        
        
        def check(removed_queries):
            temp_nums = list(nums)
            
            new_diffs = [0] * (n + 1)
            
            queries_indices = list(range(m))
            
            
            queries_indices_after_removal = sorted(list(set(range(m)) - set(removed_queries)))
            
            for i in queries_indices_after_removal:
                l, r = queries[i]
                new_diffs[l] += 1
                new_diffs[r + 1] -= 1
            
            new_prefix_sum = [0] * (n + 1)
            for i in range(n):
                new_prefix_sum[i+1] = new_prefix_sum[i] + new_diffs[i]
            
            for i in range(n):
                temp_nums[i] -= new_prefix_sum[i]
            
            
            return all(x <= 0 for x in temp_nums)

        max_removed = -1
        for i in range(1 << m):
            removed_queries = []
            for j in range(m):
                if (i >> j) & 1:
                    removed_queries.append(j)
            
            if check(removed_queries):
                if max_removed == -1 or len(removed_queries) > max_removed:
                    max_removed = len(removed_queries)
        
        return max_removed