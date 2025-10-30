class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        def encrypt(num):
            # Convert number to string to process digits
            num_str = str(num)
            # Find largest digit
            max_digit = max(num_str)
            # Replace all digits with largest digit
            return int(max_digit * len(num_str))
            
        # Encrypt each number and sum the results
        return sum(encrypt(num) for num in nums)