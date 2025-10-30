class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        def encrypt(x):
            # Convert the number to a string to iterate over digits
            digits = str(x)
            # Find the largest digit
            max_digit = max(digits)
            # Create the encrypted number by repeating the largest digit
            encrypted_number = int(max_digit * len(digits))
            return encrypted_number
        
        # Apply the encrypt function to each number in the list and sum them
        return sum(encrypt(num) for num in nums)