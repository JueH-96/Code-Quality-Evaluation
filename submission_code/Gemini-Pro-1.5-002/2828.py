class Solution:
    def smallestString(self, s: str) -> str:
        n = len(s)
        res = list(s)
        i = 0
        while i < n and res[i] == 'a':
            i += 1
        if i == n:
            res[n - 1] = 'z'
            return "".join(res)

        j = i
        while j < n and res[j] != 'a':
            res[j] = chr(ord(res[j]) - 1)
            j += 1
        
        return "".join(res)