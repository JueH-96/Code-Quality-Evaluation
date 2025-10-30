class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        ans = []
        for x, y in queries:
            max_sum = -1
            for j in range(n):
                if nums1[j] >= x and nums2[j] >= y:
                    max_sum = max(max_sum, nums1[j] + nums2[j])
            ans.append(max_sum)
        return ans