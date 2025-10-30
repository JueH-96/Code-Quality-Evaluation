class Solution:
    def smallestString(self, s: str) -> str:
        n = len(s)
        # Find the first index where s[i] != 'a'
        start = -1
        for i in range(n):
            if s[i] != 'a':
                start = i
                break
        if start == -1:
            # All 'a's
            return s[:-1] + 'z'
        else:
            # Find the end of the non-'a' block
            end = start
            while end < n and s[end] != 'a':
                end += 1
            # Modify the substring s[start:end]
            modified_sub = ''.join(chr(ord(c) - 1) for c in s[start:end])
            # Construct the result
            return s[:start] + modified_sub + s[end:]