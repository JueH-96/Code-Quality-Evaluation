class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        nums = sorted([(nums1[i], nums2[i]) for i in range(n)], key = lambda x: (-x[0], -x[1]))
        prefix = [0]*(n+1)
        for i in range(n):
            prefix[i+1] = max(prefix[i], nums[i][0] + nums[i][1])
        res = []
        for x, y in queries:
            l = 0
            r = n
            while l < r:
                mid = (l + r) // 2
                if nums[mid][0] >= x and nums[mid][1] >= y:
                    r = mid
                else:
                    l = mid + 1
            if l == n:
                res.append(-1)
            else:
                res.append(prefix[l+1])
        return res