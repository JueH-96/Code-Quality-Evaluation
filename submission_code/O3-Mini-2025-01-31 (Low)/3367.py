from typing import List

class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        total_sum = 0
        for num in nums:
            # Convert the number to a string to examine each digit.
            digits = str(num)
            # Find the maximum digit in the number.
            max_digit = max(digits)
            # Create the encrypted integer by repeating the maximum digit, 
            # with the same length as the original number.
            encrypted_num = int(max_digit * len(digits))
            # Add the encrypted number to the running total.
            total_sum += encrypted_num
        return total_sum

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.sumOfEncryptedInt([1, 2, 3]))    # Expected output: 6
    print(sol.sumOfEncryptedInt([10, 21, 31])) # Expected output: 66