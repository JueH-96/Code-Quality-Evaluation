import math
from typing import List # Required for the type hint List, usually available in competitive programming platforms.

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        
        n = len(nums)
        beautiful_pairs_count = 0

        for i in range(n):
            num_i = nums[i]
            
            # Get the first digit of nums[i].
            # Convert num_i to a string, take the first character, and convert back to int.
            # Example: if num_i = 345, str(num_i) = "345", str(num_i)[0] = "3", int("3") = 3.
            # Since nums[i] >= 1, it's always a positive number, so str(num_i)[0] is valid.
            s_num_i = str(num_i)
            first_digit_i = int(s_num_i[0])

            for j in range(i + 1, n):
                num_j = nums[j]
                
                # Get the last digit of nums[j].
                # This is simply num_j % 10.
                # Example: if num_j = 678, num_j % 10 = 8.
                # The constraint nums[k] % 10 != 0 ensures the last digit is not 0.
                last_digit_j = num_j % 10
                
                # Check if first_digit_i and last_digit_j are coprime.
                # Two integers are coprime if their greatest common divisor (GCD) is 1.
                # The digits involved are guaranteed to be in the range [1, 9].
                if math.gcd(first_digit_i, last_digit_j) == 1:
                    beautiful_pairs_count += 1
        
        return beautiful_pairs_count