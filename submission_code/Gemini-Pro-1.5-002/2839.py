class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        nums = sorted([(nums1[i], nums2[i]) for i in range(n)], key=lambda x: (-x[0], -x[1]))
        
        q = len(queries)
        queries_with_index = sorted([(queries[i][0], queries[i][1], i) for i in range(q)], key=lambda x: -x[0])
        
        ans = [-1] * q
        j = 0
        max_sum = -1
        
        for x, y, idx in queries_with_index:
            while j < n and nums[j][0] >= x:
                max_sum = max(max_sum, nums[j][0] + nums[j][1])
                j += 1
            
            k = 0
            curr_max = -1
            while k < j:
                if nums[k][1] >= y:
                    curr_max = max(curr_max, nums[k][0] + nums[k][1])
                k += 1
            
            if curr_max != -1:
                ans[idx] = curr_max
            elif max_sum != -1 and nums[0][0] >= x and nums[0][1] >= y:
                ans[idx] = max_sum
        
        return ans