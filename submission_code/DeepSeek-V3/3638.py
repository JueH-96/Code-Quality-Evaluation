from collections import Counter

class Solution:
    def makeStringGood(self, s: str) -> int:
        # Count the frequency of each character in the string
        freq = Counter(s)
        # Get the list of frequencies
        freq_list = list(freq.values())
        # Sort the frequencies in ascending order
        freq_list.sort()
        # Initialize the number of operations
        operations = 0
        # Iterate through the sorted frequencies
        for i in range(len(freq_list) - 1):
            # If the current frequency is greater than the next, we need to reduce it
            if freq_list[i] > freq_list[i+1]:
                # Calculate the difference
                diff = freq_list[i] - freq_list[i+1]
                # Add the difference to the operations
                operations += diff
                # Update the current frequency to match the next
                freq_list[i] = freq_list[i+1]
            # If the current frequency is less than the next, we need to increase it
            elif freq_list[i] < freq_list[i+1]:
                # Calculate the difference
                diff = freq_list[i+1] - freq_list[i]
                # Add the difference to the operations
                operations += diff
                # Update the next frequency to match the current
                freq_list[i+1] = freq_list[i]
        return operations