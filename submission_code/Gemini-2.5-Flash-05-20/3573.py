import collections

class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        n = len(word1)
        k = len(word2)

        # Edge case: If word2 is an empty string, any substring of word1 can have it as a prefix.
        # The total number of substrings of a string of length n is n * (n + 1) / 2.
        if k == 0:
            return n * (n + 1) // 2

        # Store the required character counts for word2
        target_counts = collections.Counter(word2)
        # Store the number of distinct characters in word2 that need to be satisfied
        num_distinct_target_chars = len(target_counts)

        current_window_counts = collections.Counter()
        left = 0
        total_valid_substrings = 0
        
        # Tracks how many distinct characters from word2 have their frequency 
        # requirement met or exceeded in the current window.
        satisfied_distinct_chars = 0 

        # Iterate with the 'right' pointer to expand the window
        for right in range(n):
            char_r = word1[right]
            current_window_counts[char_r] += 1
            
            # If char_r is a required character and its count in the window
            # just reached the target, increment satisfied_distinct_chars.
            if char_r in target_counts and current_window_counts[char_r] == target_counts[char_r]:
                satisfied_distinct_chars += 1

            # While the current window `word1[left:right+1]` satisfies the character count requirement
            while satisfied_distinct_chars == num_distinct_target_chars:
                # Calculate the maximum possible starting index `i` for a valid substring
                # that ends at `right`.
                # Condition 1: Character counts must be satisfied.
                #   If `word1[left:right+1]` satisfies char counts, then any `word1[i:right+1]`
                #   where `i <= left` also satisfies character counts (as it includes all chars from `word1[left:right+1]`).
                # Condition 2: Length `right - i + 1` must be `>= k`.
                #   This implies `i <= right - k + 1`.
                # Combining both: `i` must be less than or equal to `min(left, right - k + 1)`.
                # Also, `i` must be non-negative (`i >= 0`).
                # So, valid starting indices `i` are in the range `[0, min(left, right - k + 1)]`.
                
                # Calculate the upper bound for valid starting indices
                max_valid_start_idx = min(left, right - k + 1)
                
                # The number of valid starting indices (and thus valid substrings ending at 'right')
                # is `max_valid_start_idx - 0 + 1`, clamped to be non-negative.
                count_for_current_right = max(0, max_valid_start_idx + 1)
                total_valid_substrings += count_for_current_right

                # Shrink the window from the left
                char_l = word1[left]
                current_window_counts[char_l] -= 1
                
                # If char_l was a required character and its count in the window
                # just dropped below the target, decrement satisfied_distinct_chars.
                if char_l in target_counts and current_window_counts[char_l] == target_counts[char_l] - 1:
                    satisfied_distinct_chars -= 1
                
                left += 1 # Move the left pointer to shrink the window
                
        return total_valid_substrings