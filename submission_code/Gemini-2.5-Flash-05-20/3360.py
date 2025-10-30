import collections

class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        # 1. Count character frequencies
        counts = collections.Counter(word)
        
        # 2. Get the list of unique frequencies
        freq_list = list(counts.values())
        
        # If the word is empty, no deletions are needed.
        # Based on constraints (1 <= word.length), this check is technically not strictly necessary.
        if not freq_list:
            return 0

        # Initialize minimum deletions.
        # One possible strategy is to delete all characters, which costs the total number of characters.
        min_deletions = sum(freq_list)

        # Determine the maximum frequency of any character in the original word.
        # This helps to define the upper bound for our iteration of `R`.
        max_original_freq = max(freq_list)
        
        # Iterate through all possible values for 'R' (the maximum frequency allowed for characters
        # in the final k-special word).
        # 'R' can range from 1 (the minimum possible frequency for a character to be present)
        # up to `max_original_freq`. Values of R greater than `max_original_freq` would not change
        # the cost for any character (as all original frequencies would be <= R), so they are not needed.
        # The case where all characters are deleted (equivalent to R=0, effectively) is covered by
        # the initial `min_deletions = sum(freq_list)`.
        for R in range(1, max_original_freq + 1):
            current_deletions = 0
            
            # Calculate the lower bound 'L' for the frequencies of characters in the k-special word.
            # 'L' must satisfy two conditions:
            # 1. R - L <= k  =>  L >= R - k
            # 2. L must be at least 1, because a character must have a frequency >= 1 to be "in the string".
            L = max(1, R - k) 
            
            # Calculate deletions for the current target range [L, R]
            for f_orig in freq_list:
                if f_orig < L:
                    # If an original frequency is less than the minimum allowed frequency `L`,
                    # all occurrences of this character must be deleted.
                    current_deletions += f_orig
                elif f_orig > R:
                    # If an original frequency is greater than the maximum allowed frequency `R`,
                    # characters must be deleted to bring its frequency down to `R`.
                    current_deletions += (f_orig - R)
                # If L <= f_orig <= R, the character's frequency is already within the desired range.
                # To minimize deletions, we keep all occurrences of this character (cost 0).
            
            # Update the overall minimum deletions found so far
            min_deletions = min(min_deletions, current_deletions)
            
        return min_deletions