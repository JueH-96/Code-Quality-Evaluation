class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        bit_count = [0] * 31
        for num in nums:
            for i in range(31):
                if num & (1 << i):
                    bit_count[i] += 1
        result = 0
        for i in range(31):
            if bit_count[i] >= k:
                result |= (1 << i)
        return result