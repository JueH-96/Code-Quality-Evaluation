class Solution:
    def makeStringGood(self, s: str) -> int:
        from collections import Counter
        
        # Count the frequency of each character in the string
        freq = Counter(s)
        
        # Get the frequencies in a list
        freq_values = list(freq.values())
        
        # Find the minimum and maximum frequency
        min_freq = min(freq_values)
        max_freq = max(freq_values)
        
        # If all frequencies are already the same, return 0
        if min_freq == max_freq:
            return 0
        
        # Calculate the total number of characters
        total_chars = sum(freq_values)
        
        # Calculate the number of distinct characters
        distinct_chars = len(freq_values)
        
        # Calculate the target frequency for each character to make the string good
        target_freq = total_chars // distinct_chars
        
        # If the target frequency is less than the minimum frequency, we need to increase it
        if target_freq < min_freq:
            target_freq = min_freq
        
        # Calculate the number of operations needed to make all frequencies equal to target_freq
        operations = 0
        for f in freq_values:
            if f > target_freq:
                operations += f - target_freq
            elif f < target_freq:
                operations += target_freq - f
        
        return operations