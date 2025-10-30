class Solution:
    def makeStringGood(self, s: str) -> int:
        # Count frequencies of each character in the string
        freq = {}
        for char in s:
            freq[char] = freq.get(char, 0) + 1
        
        min_operations = float('inf')
        
        # Try all possible target frequencies from 0 to max frequency
        for target_freq in range(max(freq.values()) + 1):
            operations = 0
            excess = 0  # Tracks excess from previous character
            
            # Iterate through all lowercase letters in order
            for char in "abcdefghijklmnopqrstuvwxyz":
                current_freq = freq.get(char, 0)
                
                if current_freq > target_freq:
                    # We have too many of this character
                    char_excess = current_freq - target_freq
                    # Use previous excess to convert to this character
                    operations_needed = char_excess - min(excess, char_excess)
                    operations += operations_needed
                    # Remaining excess after using some for operations
                    excess = char_excess - operations_needed
                elif current_freq < target_freq:
                    # We have too few of this character
                    deficit = target_freq - current_freq
                    # Use change operations from previous character if possible
                    transfer = min(excess, deficit)
                    excess -= transfer
                    # Fill remaining deficit with insertions
                    operations += deficit - transfer
                # If current_freq == target_freq, no operations needed
                
            min_operations = min(min_operations, operations)
        
        return min_operations