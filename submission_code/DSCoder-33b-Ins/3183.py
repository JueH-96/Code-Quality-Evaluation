class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        res = 0
        for i in range(31, -1, -1):
            count = sum((num >> i) & 1 for num in nums)
            if count >= k:
                res += 1 << i
        return res