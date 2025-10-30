class Solution:
    def smallestString(self, s: str) -> str:
        res = list(s)
        for i in range(len(s)):
            for j in range(i, len(s)):
                temp = list(s)
                for k in range(i, j + 1):
                    if temp[k] == 'a':
                        temp[k] = 'z'
                    else:
                        temp[k] = chr(ord(temp[k]) - 1)
                temp = ''.join(temp)
                if temp < ''.join(res):
                    res = list(temp)
        return ''.join(res)