class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        def encrypt(x):
            x_str = str(x)
            max_digit = max(x_str)
            return int(max_digit * len(x_str))
        
        encrypted_sum = 0
        for num in nums:
            encrypted_sum += encrypt(num)
        return encrypted_sum