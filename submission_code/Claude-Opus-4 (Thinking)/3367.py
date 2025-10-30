class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        return sum(int(max(str(num)) * len(str(num))) for num in nums)