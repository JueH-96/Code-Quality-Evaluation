class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        pairs = sorted(zip(nums1, nums2), reverse=True)
        
        ans = [-1] * len(queries)
        
        for i, (x, y) in enumerate(queries):
            max_sum = -1
            for num1, num2 in pairs:
                if num1 >= x and num2 >= y:
                    max_sum = num1 + num2
                    break
            ans[i] = max_sum
        
        return ans