class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        from collections import Counter
        
        # Count frequency of each character in the word
        freq = Counter(word)
        # Get the list of frequencies
        freq_values = list(freq.values())
        freq_values.sort()
        
        # Total number of characters in the word
        total_chars = len(word)
        
        # Initialize minimum deletions to a large number
        min_deletions = float('inf')
        
        # Try to make the word k-special by targeting each possible frequency
        for target_freq in range(max(1, min(freq_values)), max(freq_values) + 1):
            current_deletions = 0
            current_count = 0
            
            # Calculate deletions required to make all frequencies either
            # within the range [target_freq, target_freq + k]
            # or reduce them to zero
            for f in freq_values:
                if f < target_freq:
                    current_deletions += f  # Remove all characters of this frequency
                elif f > target_freq + k:
                    current_deletions += (f - (target_freq + k))  # Reduce to target_freq + k
                else:
                    current_count += f  # This frequency is already within the range
            
            # Update the minimum deletions found so far
            min_deletions = min(min_deletions, total_chars - current_count)
        
        return min_deletions