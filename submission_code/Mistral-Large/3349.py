class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        from collections import defaultdict

        # Initialize variables
        left = 0
        max_length = 0
        char_count = defaultdict(int)

        # Iterate over the string with the right pointer
        for right in range(len(s)):
            char_count[s[right]] += 1

            # If any character count exceeds 2, move the left pointer
            while any(count > 2 for count in char_count.values()):
                char_count[s[left]] -= 1
                if char_count[s[left]] == 0:
                    del char_count[s[left]]
                left += 1

            # Calculate the maximum length of the valid substring
            max_length = max(max_length, right - left + 1)

        return max_length