class Solution:
    def smallestString(self, s: str) -> str:
        n = len(s)
        start = -1
        for i in range(n):
            if s[i] != 'a':
                start = i
                break
        if start == -1:
            return 'z' * n
        else:
            end = start
            while end < n and s[end] != 'a':
                end += 1
            end -= 1  # adjust to the last non 'a' character in the run
            res = list(s)
            for i in range(start, end + 1):
                c = res[i]
                res[i] = chr((ord(c) - ord('a') - 1) % 26 + ord('a'))
            return ''.join(res)