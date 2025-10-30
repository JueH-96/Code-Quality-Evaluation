class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        def encrypt(x):
            str_x = str(x)
            max_digit = max(str_x)
            encrypted = int(max_digit * len(str_x))
            return encrypted
        
        return sum(encrypt(num) for num in nums)