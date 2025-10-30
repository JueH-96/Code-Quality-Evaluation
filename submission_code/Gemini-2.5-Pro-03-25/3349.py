import collections

class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        """
        Finds the maximum length substring of s such that each character
        appears at most twice within the substring. Uses a sliding window approach.

        Args:
            s: The input string (lowercase English letters, length >= 2).

        Returns:
            The maximum length of a valid substring.
        """
        n = len(s)
        # Constraints state n >= 2, so we don't need explicit checks for n=0 or n=1.
        
        max_len = 0  # Initialize maximum length found
        left = 0     # Left pointer of the sliding window (inclusive)
        
        # Frequency map (dictionary) to store counts of characters 
        # in the current window s[left...right].
        # collections.defaultdict(int) initializes counts to 0 automatically.
        counts = collections.defaultdict(int) 

        # Iterate through the string with the right pointer, expanding the window
        for right in range(n):
            char_right = s[right]  # The character entering the window
            counts[char_right] += 1 # Increment its count in the window

            # Check if the condition (at most two occurrences) is violated 
            # specifically by the character just added. If its count is now > 2,
            # we need to shrink the window from the left until this character's
            # count is back down to 2.
            while counts[char_right] > 2:
                char_left = s[left]    # The character leaving the window from the left
                counts[char_left] -= 1 # Decrement its count
                
                # Optional optimization: remove the key if count becomes zero.
                # Not strictly necessary with defaultdict, but can keep dict size smaller.
                # if counts[char_left] == 0:
                #     del counts[char_left]
                     
                left += 1              # Move the left pointer forward, shrinking the window

            # After the while loop, the current window s[left...right] is guaranteed 
            # to be valid (all character counts are <= 2).
            # Calculate the length of this valid window.
            current_len = right - left + 1
            
            # Update the overall maximum length found so far.
            max_len = max(max_len, current_len)

        # Return the maximum length found across all valid windows.
        return max_len