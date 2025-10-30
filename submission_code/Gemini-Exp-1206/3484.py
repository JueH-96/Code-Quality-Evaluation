class Solution:
    def getSmallestString(self, s: str) -> str:
        s = list(s)
        n = len(s)
        for i in range(n - 1):
            min_idx = i
            for j in range(i + 1, n):
                if int(s[j]) % 2 == int(s[i]) % 2:
                    if s[j] < s[min_idx]:
                        min_idx = j
            
            if min_idx != i:
                
                if int(s[min_idx]) % 2 == int(s[i]) % 2:
                    
                    for k in range(min_idx, i, -1):
                        if int(s[k]) % 2 == int(s[k-1]) % 2:
                            s[k], s[k-1] = s[k-1], s[k]
                        else:
                            break
                    break
        return "".join(s)