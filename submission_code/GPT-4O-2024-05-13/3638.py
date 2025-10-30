from collections import Counter

class Solution:
    def makeStringGood(self, s: str) -> int:
        # Count the frequency of each character in the string
        freq = Counter(s)
        
        # Get the list of frequencies
        freq_values = list(freq.values())
        
        # Find the most common frequency
        most_common_freq = max(freq_values, key=freq_values.count)
        
        # Calculate the number of operations needed to make all frequencies equal to the most common frequency
        operations = 0
        for f in freq_values:
            if f != most_common_freq:
                operations += abs(f - most_common_freq)
        
        return operations