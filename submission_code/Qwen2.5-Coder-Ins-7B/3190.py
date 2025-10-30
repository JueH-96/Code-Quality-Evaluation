class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        if nums1[-1] > max(nums1) or nums2[-1] > max(nums2):
            return -1
        
        def count_swaps(keep_last):
            swaps = 0
            for i in range(n - 1):
                if (nums1[i] > nums1[-1] or nums2[i] > nums2[-1]) and (nums1[i] > nums2[-1] or nums2[i] > nums1[-1]):
                    if keep_last:
                        return float('inf')
                    else:
                        swaps += 1
            return swaps
        
        keep_last = count_swaps(True)
        swap_last = count_swaps(False) + 1
        
        return min(keep_last, swap_last)