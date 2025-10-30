class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        ans = 0
        for i in range(n):
            for j in range(i, n):
                sub = s[i:j+1]
                counts = {}
                for char in sub:
                    counts[char] = counts.get(char, 0) + 1
                
                valid = False
                for char in counts:
                    if counts[char] >= k:
                        valid = True
                        break
                
                if valid:
                    ans += 1
        return ans