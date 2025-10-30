class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        count = 0
        for i in range(n):
            for j in range(i, n):
                sub = s[i:j+1]
                freq = {}
                for char in sub:
                    freq[char] = freq.get(char, 0) + 1
                
                valid = False
                for char_freq in freq.values():
                    if char_freq >= k:
                        valid = True
                        break
                if valid:
                    count += 1
        return count