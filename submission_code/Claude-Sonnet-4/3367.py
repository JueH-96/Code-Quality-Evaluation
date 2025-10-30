class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        def encrypt(x):
            # Convert to string to work with digits
            str_x = str(x)
            # Find the maximum digit
            max_digit = max(str_x)
            # Create encrypted number by repeating max digit
            encrypted_str = max_digit * len(str_x)
            return int(encrypted_str)
        
        # Apply encryption to each number and sum them
        return sum(encrypt(num) for num in nums)