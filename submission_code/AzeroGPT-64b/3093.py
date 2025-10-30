class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        sum_ = 0
        for index, num in enumerate(nums):
            binary = bin(index)[2:]
            if binary.count('1') == k:
                sum_ += num
        return sum_