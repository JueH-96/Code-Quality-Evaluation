class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        ans = 0
        for i in range(n):
            for j in range(i, n):
                sub = word[i:j+1]
                len_sub = len(sub)
                if len_sub % k != 0:
                    continue
                
                counts = {}
                for char in sub:
                    counts[char] = counts.get(char, 0) + 1
                
                valid = True
                for char in counts:
                    if counts[char] != k:
                        valid = False
                        break
                if not valid:
                    continue
                
                valid = True
                for l in range(len_sub - 1):
                    if abs(ord(sub[l]) - ord(sub[l+1])) > 2:
                        valid = False
                        break
                if valid:
                    ans += 1
        return ans