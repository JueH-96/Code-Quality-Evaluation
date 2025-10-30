class Solution:
    def maximumLength(self, s: str) -> int:
        # Initialize the maximum length to -1, assuming no special substring occurs thrice
        max_length = -1

        # Iterate through the string to find special substrings
        for char in set(s):  # Using set to avoid duplicate characters
            # Count occurrences of the character
            count = s.count(char)
            if count >= 3:
                # If the character occurs at least thrice, find the maximum length of the special substring
                for length in range(count, 2, -1):
                    # Check if the substring of the given length occurs at least thrice
                    if s.count(char * length) >= 3:
                        # Update the maximum length if a longer special substring is found
                        max_length = max(max_length, length)
                        break  # No need to check shorter lengths for this character

        return max_length