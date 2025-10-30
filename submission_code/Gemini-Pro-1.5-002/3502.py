class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        count = 0
        for i in range(n):
            for j in range(i, n):
                sub = s[i:j+1]
                freq = {}
                valid = False
                for char in sub:
                    freq[char] = freq.get(char, 0) + 1
                for char in freq:
                    if freq[char] >= k:
                        valid = True
                        break
                if valid:
                    count += 1
        return count