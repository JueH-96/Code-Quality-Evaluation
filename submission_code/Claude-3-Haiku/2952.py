class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        time = 0
        while True:
            total = sum(nums1)
            if total <= x:
                return time
            if total > x:
                min_idx = -1
                min_val = float('inf')
                for i in range(n):
                    nums1[i] += nums2[i]
                    if nums1[i] < min_val:
                        min_idx = i
                        min_val = nums1[i]
                nums1[min_idx] = 0
                time += 1
            if total == x:
                return time