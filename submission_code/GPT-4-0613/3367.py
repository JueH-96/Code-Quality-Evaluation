class Solution:
    def sumOfEncryptedInt(self, nums):
        def encrypt(x):
            max_digit = max(int(d) for d in str(x))
            return int(str(max_digit) * len(str(x)))

        return sum(encrypt(x) for x in nums)