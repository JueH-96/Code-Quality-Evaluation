class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        count = 0
        n = len(s)
        if n == 0 or c not in s:
            return 0

        indices = [i for i, x in enumerate(s) if x == c]
        
        if len(indices) == 0:
            return 0
        
        for i in range(len(indices)):
            for j in range(i, len(indices)):
                substring = s[indices[i]:indices[j]+1]
                if substring[0] == c and substring[-1] == c:
                    count +=1
        return count