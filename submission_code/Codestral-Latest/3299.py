class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        from collections import Counter

        # Count the frequency of each number in the array
        freq = Counter(nums)

        # Initialize the maximum length of the subset
        max_length = 0

        # Iterate through each unique number in the array
        for num in freq:
            # Initialize the current length of the subset
            current_length = 0

            # Check if the number can be part of the subset
            if freq[num] > 0:
                current_length += freq[num]

                # Check for the next power of 2
                power = 2
                while freq[num * power] > 0:
                    current_length += freq[num * power]
                    power *= 2

                # Check for the previous power of 2
                power = 2
                while freq[num // power] > 0:
                    current_length += freq[num // power]
                    power *= 2

                # Update the maximum length of the subset
                max_length = max(max_length, current_length)

        return max_length