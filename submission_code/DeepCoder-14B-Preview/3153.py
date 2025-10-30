class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        S = sum(nums)
        max_or = 0
        for num in nums:
            max_or |= num
        t = min(k, S // max_or) if max_or != 0 else 0
        if t < k:
            sum_sq = t * (max_or ** 2) + (S - t * max_or) ** 2
        else:
            sum_sq = t * (max_or ** 2)
        return sum_sq % (10**9 + 7)