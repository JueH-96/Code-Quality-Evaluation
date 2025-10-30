class Solution:
    def maximumLength(self, s: str) -> int:
        from collections import defaultdict

        # Dictionary to store the count of each special substring
        substring_count = defaultdict(int)

        # Length of the string
        n = len(s)

        # Generate all possible special substrings
        for i in range(n):
            for j in range(i + 1, n + 1):
                substring = s[i:j]
                # Check if the substring is special
                if len(set(substring)) == 1:
                    substring_count[substring] += 1

        # Find the longest special substring that occurs at least thrice
        max_length = -1
        for substring, count in substring_count.items():
            if count >= 3:
                max_length = max(max_length, len(substring))

        return max_length