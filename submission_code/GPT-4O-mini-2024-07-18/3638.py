class Solution:
    def makeStringGood(self, s: str) -> int:
        from collections import Counter
        
        # Count the frequency of each character in the string
        freq = Counter(s)
        
        # Get the frequency of the frequencies
        freq_count = Counter(freq.values())
        
        # The minimum number of operations needed
        operations = 0
        
        # We need to find the maximum frequency that we can use to make the string good
        max_freq = max(freq_count.keys())
        
        # Check for each frequency from 1 to max_freq
        for target_freq in range(1, max_freq + 1):
            total_operations = 0
            
            # Calculate how many operations are needed to make all characters have target_freq
            for char_freq, count in freq_count.items():
                if char_freq < target_freq:
                    # Need to insert characters to reach target_freq
                    total_operations += (target_freq - char_freq) * count
                elif char_freq > target_freq:
                    # Need to delete characters to reduce to target_freq
                    total_operations += (char_freq - target_freq) * count
            
            # Update the minimum operations needed
            if operations == 0 or total_operations < operations:
                operations = total_operations
        
        return operations