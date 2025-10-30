class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        total = (1 << 32) - 1
        for num in nums:
            total &= num
        if total != 0:
            return 1
        count = 0
        cur = -1
        for num in nums:
            cur &= num
            if cur == 0:
                count += 1
                cur = -1
        return count