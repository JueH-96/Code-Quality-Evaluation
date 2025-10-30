import math

class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        min_len = math.inf
        min_substring = ""
        one_count = 0
        left = 0

        for right in range(len(s)):
            if s[right] == '1':
                one_count += 1

            # While condition: s[left...right] is a beautiful substring (has exactly k ones)
            # We keep shrinking the window from the left as long as it remains beautiful
            while one_count == k:
                current_substring = s[left:right+1]
                current_len = right - left + 1

                # If we found a new shortest beautiful substring
                if current_len < min_len:
                    min_len = current_len
                    min_substring = current_substring
                # If we found a beautiful substring with the current minimum length
                elif current_len == min_len:
                    # Compare lexicographically
                    # Since min_len is not inf at this point, min_substring must be non-empty
                    if current_substring < min_substring:
                         min_substring = current_substring

                # Try to shrink the window from the left
                # If s[left] is '1', removing it will make one_count < k, exiting the while loop
                # If s[left] is '0', removing it keeps one_count == k, and the loop continues with a shorter window
                if s[left] == '1':
                    one_count -= 1
                left += 1

        # If min_len is still inf, no beautiful substring was found.
        # min_substring remains "", which is the desired output.
        return min_substring