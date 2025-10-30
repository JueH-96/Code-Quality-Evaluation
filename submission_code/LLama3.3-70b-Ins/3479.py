class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        count = 0
        
        for i in range(n):
            for j in range(i, n):
                substring = s[i:j+1]
                zeros = substring.count('0')
                ones = substring.count('1')
                
                if ones >= zeros ** 2:
                    count += 1
                    
        return count