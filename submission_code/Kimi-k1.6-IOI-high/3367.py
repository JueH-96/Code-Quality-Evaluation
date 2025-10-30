class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        total = 0
        for num in nums:
            s = str(num)
            max_digit = max(s)
            encrypted = max_digit * len(s)
            total += int(encrypted)
        return total