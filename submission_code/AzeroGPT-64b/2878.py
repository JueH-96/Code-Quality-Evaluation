class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        prefix, min_prefix = 0, 0
        for i, num in enumerate(nums):
            prefix += num
            if i >= k:
                min_prefix = min(min_prefix, prefix)
            if i >= k - 1 and prefix - min_prefix == 0:
                prefix -= nums[i - k + 1]
                min_prefix = max(min_prefix, 0)
            elif prefix - min_prefix > 0:
                return False
        return True