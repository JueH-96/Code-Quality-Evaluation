import collections

class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        """
        Calculates the maximum length of a substring with at most two occurrences of each character.

        This implementation uses a sliding window approach for optimal performance.
        A window, defined by 'left' and 'right' pointers, slides over the string.
        A frequency map keeps track of character counts within the window. The window
        expands to the right and shrinks from the left whenever a character's count
        exceeds two, ensuring the substring within the window is always valid.
        """
        
        # A frequency map to store counts of characters in the current window.
        # defaultdict(int) initializes new keys with a value of 0.
        char_counts = collections.defaultdict(int)
        
        max_len = 0
        left = 0
        
        # The 'right' pointer iterates through the string, expanding the window.
        for right in range(len(s)):
            # Add the new character to the window and update its count.
            char_right = s[right]
            char_counts[char_right] += 1
            
            # If the count of the newly added character exceeds 2, the window is invalid.
            # Shrink the window from the left until it becomes valid again.
            while char_counts[char_right] > 2:
                char_left = s[left]
                char_counts[char_left] -= 1
                left += 1
                
            # The current window s[left:right+1] is valid.
            # Update the maximum length found so far.
            max_len = max(max_len, right - left + 1)
            
        return max_len