class Solution:
    def smallestString(self, s: str) -> str:
        result = list(s)
        for i in range(len(s)):
            for j in range(i, len(s)):
                temp = result[:]
                for k in range(j, len(s)):
                    temp[k] = chr(ord(temp[k]) - 1) if temp[k] != 'a' else 'z'
                if ''.join(temp) < ''.join(result):
                    result = temp
        return ''.join(result)