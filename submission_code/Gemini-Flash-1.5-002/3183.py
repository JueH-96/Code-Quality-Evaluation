class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        result = 0
        for bit in range(31):
            count = 0
            for num in nums:
                if (num >> bit) & 1:
                    count += 1
            if count >= k:
                result |= (1 << bit)
        return result