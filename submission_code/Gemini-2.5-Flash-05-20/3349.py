import collections

class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        max_len = 0
        left = 0
        
        # Use a dictionary to store character frequencies within the current window
        # collections.defaultdict(int) is convenient for automatically
        # initializing new character counts to 0.
        char_counts = collections.defaultdict(int)
        
        # Iterate with the right pointer to expand the window
        for right in range(len(s)):
            current_char = s[right]
            
            # Add the current character to the window and increment its count
            char_counts[current_char] += 1
            
            # Check if the current window violates the condition (any character count > 2).
            # If the count of the `current_char` (s[right]) just exceeded 2,
            # we need to shrink the window from the left.
            # It's sufficient to check only `current_char` because if any
            # other character's count was already > 2, the `while` loop
            # would have already processed it in a previous iteration.
            while char_counts[current_char] > 2:
                char_to_remove = s[left]
                char_counts[char_to_remove] -= 1
                left += 1
            
            # At this point, the substring s[left:right+1] is valid,
            # meaning all characters within it appear at most twice.
            # Calculate its length and update the maximum length found so far.
            max_len = max(max_len, right - left + 1)
            
        return max_len