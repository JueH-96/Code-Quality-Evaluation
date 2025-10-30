class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        xor = 0
        for num in nums:
            xor ^= num
        return xor ^ reduce(xor.__and__, nums)