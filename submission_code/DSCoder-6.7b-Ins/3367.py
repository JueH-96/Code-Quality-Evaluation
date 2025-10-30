class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        def encrypt(x):
            return int(str(x).replace(max(str(x)), '9'))
        return sum(encrypt(x) for x in nums)