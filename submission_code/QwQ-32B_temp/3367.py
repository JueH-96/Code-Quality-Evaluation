class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        total = 0
        for num in nums:
            s = str(num)
            max_digit = int(max(s))
            length = len(s)
            encrypted = int(str(max_digit) * length)
            total += encrypted
        return total