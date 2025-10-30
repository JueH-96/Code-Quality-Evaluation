class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        
        def count_ops(swap_last):
            last1 = nums2[-1] if swap_last else nums1[-1]
            last2 = nums1[-1] if swap_last else nums2[-1]
            ops = 1 if swap_last else 0
            for i in range(n - 1):
                a, b = nums1[i], nums2[i]
                if a <= last1 and b <= last2:
                    continue
                if b <= last1 and a <= last2:
                    ops += 1
                else:
                    return float('inf')
            return ops
        
        result1 = count_ops(False)
        result2 = count_ops(True)
        ans = min(result1, result2)
        return ans if ans != float('inf') else -1