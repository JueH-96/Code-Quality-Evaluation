class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        max_bit = 0
        for num in nums:
            if num > 0:
                max_bit = max(max_bit, num.bit_length())
        k_or = 0
        for i in range(max_bit):
            count = 0
            for num in nums:
                if (num >> i) & 1:
                    count += 1
            if count >= k:
                k_or |= (1 << i)
        return k_or