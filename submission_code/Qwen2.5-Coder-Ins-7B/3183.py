class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        result = 0
        for i in range(32):
            count = sum((num >> i) & 1 for num in nums)
            if count >= k:
                result |= (1 << i)
        return result