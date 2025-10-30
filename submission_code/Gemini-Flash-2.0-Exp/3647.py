class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        m = len(queries)
        
        def check(removed_queries):
            temp_nums = nums[:]
            used_queries = []
            for i in range(m):
                if i not in removed_queries:
                    used_queries.append(queries[i])
            
            for query in used_queries:
                l, r = query
                for i in range(l, r + 1):
                    if temp_nums[i] > 0:
                        temp_nums[i] -= 1
            
            return all(x == 0 for x in temp_nums)
        
        
        max_removed = -1
        for i in range(1 << m):
            removed_queries = []
            count = 0
            for j in range(m):
                if (i >> j) & 1:
                    removed_queries.append(j)
                    count += 1
            
            if check(removed_queries):
                max_removed = max(max_removed, count)
        
        if max_removed == -1 and check([]):
            return 0
        elif max_removed == -1:
            return -1
        else:
            return max_removed