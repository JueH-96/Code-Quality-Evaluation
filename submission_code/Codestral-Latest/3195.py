class Solution:
    def minimumSteps(self, s: str) -> int:
        # Count the number of '1's in the string
        count_ones = s.count('1')

        # Initialize a variable to keep track of the number of '1's encountered so far
        ones_encountered = 0

        # Initialize a variable to keep track of the minimum steps required
        min_steps = 0

        # Iterate through the string
        for char in s:
            if char == '1':
                ones_encountered += 1
            else:
                # If we encounter a '0', we need to swap it with a '1' to the right
                # The number of swaps required is the number of '1's to the right of the current position
                min_steps += count_ones - ones_encountered

        return min_steps