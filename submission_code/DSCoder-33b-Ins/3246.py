class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        def count_trailing_zeros(n):
            binary = bin(n)[2:]
            return len(binary) - len(binary.rstrip('0'))

        n = len(nums)
        for i in range(1, 2**n):
            bitwise_or = nums[0]
            for j in range(1, n):
                if (i >> j) & 1:
                    bitwise_or |= nums[j]
            if count_trailing_zeros(bitwise_or) > 0:
                return True
        return False