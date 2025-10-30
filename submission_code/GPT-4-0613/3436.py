class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        min_diff = float('inf')
        for i in range(n):
            or_val = nums[i]
            for j in range(i, n):
                or_val |= nums[j]
                min_diff = min(min_diff, abs(k - or_val))
                if min_diff == 0:
                    return 0
                if or_val > k:
                    break
        return min_diff