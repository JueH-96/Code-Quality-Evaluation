class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        
        def calculate_swaps(last1, last2):
            swaps = 0
            for i in range(n-1):
                a, b = nums1[i], nums2[i]
                cond1 = (a <= last1) and (b <= last2)
                cond2 = (b <= last1) and (a <= last2)
                if not cond1 and not cond2:
                    return float('inf')
                if not cond1:
                    swaps += 1
            return swaps
        
        option1 = calculate_swaps(nums1[-1], nums2[-1])
        option2 = 1 + calculate_swaps(nums2[-1], nums1[-1])
        
        ans = min(option1, option2)
        return ans if ans != float('inf') else -1