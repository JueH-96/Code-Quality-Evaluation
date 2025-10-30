class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)

        def solve(target1, target2):
            ops = 0
            for i in range(n - 1):
                if nums1[i] > target1 or nums2[i] > target2:
                    if nums1[i] > target2 or nums2[i] > target1:
                        return float('inf')
                    ops += 1
            return ops

        ans = min(solve(nums1[-1], nums2[-1]), solve(nums2[-1], nums1[-1]) + 1)

        if ans == float('inf'):
            return -1
        else:
            return ans