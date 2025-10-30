class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        total = 0
        for x in nums:
            s = str(x)
            max_digit = max(s)
            encrypted = int(max_digit * len(s))
            total += encrypted
        return total