class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        total_sum = sum(nums)
        
        
        def check(removed_count):
            temp_nums = list(nums)
            used_queries = [True] * len(queries)
            
            
            sorted_queries = sorted(range(len(queries)), key=lambda i: (queries[i][1]-queries[i][0],queries[i][0]))
            
            removed_indices = []
            count = 0
            for i in sorted_queries:
                if count < removed_count:
                    used_queries[i] = False
                    count +=1
                
            
            for i in range(len(queries)):
                if used_queries[i]:
                    l, r = queries[i]
                    for j in range(l, r + 1):
                        dec = min(temp_nums[j],1)
                        temp_nums[j] -= dec
                        total_sum -= dec

            return all(x == 0 for x in temp_nums)

        left, right = 0, len(queries)
        ans = -1
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
        return ans