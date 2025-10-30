class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        bit = [0] * 31
        for num in nums:
            for i in range(30, -1, -1):
                if num & (1 << i):
                    bit[i] += 1
        res = 0
        for _ in range(k):
            idx = bit.index(max(bit))
            res |= 1 << idx
            bit[idx] = n - sum(bit[:idx])
        return res