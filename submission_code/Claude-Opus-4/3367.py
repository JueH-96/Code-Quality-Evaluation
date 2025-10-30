class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        def encrypt(x):
            # Convert to string to easily access digits
            str_x = str(x)
            # Find the maximum digit
            max_digit = max(str_x)
            # Replace all digits with the max digit
            encrypted = max_digit * len(str_x)
            return int(encrypted)
        
        # Sum all encrypted values
        return sum(encrypt(num) for num in nums)