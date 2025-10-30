class Solution:
    def smallestString(self, s: str) -> str:
        result = list(s)
        for i in range(len(s)):
            if result[i] != 'a':
                result[i] = chr(ord(result[i]) - 1)
                break
        return ''.join(result)