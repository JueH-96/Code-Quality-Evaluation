class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        ans = []
        for x_i, y_i in queries:
            max_sum = -1
            found = False
            for j in range(n):
                if nums1[j] >= x_i and nums2[j] >= y_i:
                    current_sum = nums1[j] + nums2[j]
                    max_sum = max(max_sum, current_sum)
                    found = True
            ans.append(max_sum)
        return ans