class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        pairs = [(nums1[i], nums2[i]) for i in range(n)]
        pairs.sort(key=lambda p: p[1])

        total_sum = sum(nums1)
        if total_sum <= x:
            return 0

        current_sum = 0
        time = 0

        for i in range(n):
            time += 1
            current_sum += pairs[i][0] + pairs[i][1] * time
            total_sum += pairs[i][1]
            if total_sum - current_sum <= x:
                return time

        return -1