class Solution:
    def smallestString(self, s: str) -> str:
        n = len(s)

        # Check if the entire string can be converted
        if any(char != 'a' for char in s):
            return ''.join(chr((ord(char) - ord('a') - 1) % 26 + ord('a')) for char in s)

        # Find the longest prefix of 'a'
        for i in range(n):
            if s[i] != 'a':
                break

        # Convert the first non-'a' character and all characters before it
        if i == n - 1 and s[i] == 'a':
            return s[:-1] + 'z'
        else:
            return s[:i] + chr((ord(s[i]) - ord('a') - 1) % 26 + ord('a')) + s[i+1:]