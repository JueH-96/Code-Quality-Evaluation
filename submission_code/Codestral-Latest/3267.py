class Solution:
    def maximumLength(self, s: str) -> int:
        from collections import defaultdict

        # Dictionary to store the count of each character
        char_count = defaultdict(int)

        # Variable to store the length of the longest special substring
        max_length = -1

        # Iterate through the string and count the occurrences of each character
        for char in s:
            char_count[char] += 1

        # Iterate through the dictionary to find the longest special substring
        for char, count in char_count.items():
            if count >= 3:
                max_length = max(max_length, 1)

        return max_length