class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        answer = []
        
        for x, y in queries:
            max_sum = -1
            for i in range(n):
                if nums1[i] >= x and nums2[i] >= y:
                    max_sum = max(max_sum, nums1[i] + nums2[i])
            answer.append(max_sum)
        
        return answer