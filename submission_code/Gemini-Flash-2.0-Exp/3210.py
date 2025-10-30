class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        count = 0
        for i in range(n):
            for j in range(i, n):
                sub = s[i:j+1]
                vowels = 0
                consonants = 0
                for char in sub:
                    if char in "aeiou":
                        vowels += 1
                    else:
                        consonants += 1
                if vowels == consonants and (vowels * consonants) % k == 0:
                    count += 1
        return count