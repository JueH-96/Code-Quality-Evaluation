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
        # Iterate through the frequencies and adjust them to be the same
        for i in range(len(freq_list) - 1):
            if freq_list[i] == freq_list[i+1]:
                continue
            else:
                # Calculate the difference and add to operations
                operations += abs(freq_list[i] - freq_list[i+1])
                # Set the current frequency to the next one to make them equal
                freq_list[i+1] = freq_list[i]
        return operations