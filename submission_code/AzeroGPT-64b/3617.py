from collections import Counter

class Solution:
    def possibleStringCount(self, word: str) -> int:
        counts = Counter(word)
        excess_chars = [ch for ch, count in counts.items() if count > 1]
        
        if not excess_chars:
            return 1
        
        # Determine the character with the highest count which is not at the start
        max_excess_char = max(excess_chars, key=lambda ch: counts[ch])
        # Calculate the number of possible strings by removing the character with the highest count
        # If it's not at the start, we consider each position it can be removed from
        removal_indices = [i for i, c in enumerate(word) if c == max_excess_char and i != 0]
        result = len(removal_indices) + 1  # +1 for the original string
        
        return result