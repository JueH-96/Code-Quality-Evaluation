from typing import List

class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        # Use a frequency array to count occurrences of each number.
        # Since the maximum value of nums[i] is 50, we need an array
        # of size 51 to hold counts for numbers 1 through 50.
        # Index 0 will not be used as nums[i] >= 1.
        counts = [0] * 51

        # Iterate through the input array nums and count occurrences.
        for num in nums:
            # Increment the count for the current number.
            # The number itself is used as the index into the counts array.
            counts[num] += 1

        # Initialize the result variable for the bitwise XOR sum.
        # Start with 0, as XORing with 0 does not change the value.
        xor_sum = 0

        # Iterate through the possible number values (from 1 to 50)
        # and check their counts.
        for i in range(1, 51):
            # If the count for number 'i' is exactly 2, it means
            # the number appeared twice in the input array.
            if counts[i] == 2:
                # Perform bitwise XOR with the number 'i'.
                xor_sum ^= i

        # Return the final XOR sum of all numbers that appeared twice.
        return xor_sum