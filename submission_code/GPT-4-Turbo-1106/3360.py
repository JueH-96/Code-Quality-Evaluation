class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        from collections import Counter
        
        # Count the frequency of each character in the word
        freq = Counter(word)
        
        # Sort the frequencies in descending order
        freq_values = sorted(freq.values(), reverse=True)
        
        # Initialize the minimum deletions to zero
        min_deletions = 0
        
        # Iterate over the frequencies
        for i in range(len(freq_values)):
            # If the current frequency is within the allowed range, continue
            if i == 0 or freq_values[i] + k >= freq_values[i - 1]:
                continue
            # Otherwise, calculate the deletions needed to bring it within the range
            else:
                deletions_needed = freq_values[i - 1] - k
                if deletions_needed > freq_values[i]:
                    min_deletions += freq_values[i]
                    freq_values[i] = 0
                else:
                    min_deletions += deletions_needed
                    freq_values[i] -= deletions_needed
        
        return min_deletions