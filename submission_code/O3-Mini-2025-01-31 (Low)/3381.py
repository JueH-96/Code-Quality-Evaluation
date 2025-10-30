class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        res = float('inf')
        for i in range(n):
            or_val = 0
            for j in range(i, n):
                or_val |= nums[j]
                if or_val >= k:
                    res = min(res, j - i + 1)
                    break  # no need to extend further, as we found a valid subarray starting from i
        return res if res != float('inf') else -1