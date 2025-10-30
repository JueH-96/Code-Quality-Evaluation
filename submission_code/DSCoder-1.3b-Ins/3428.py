class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        xor_result = 0
        for num in nums:
            xor_result ^= num
        return xor_result