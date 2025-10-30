class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        vowels_set = set(['a', 'e', 'i', 'o', 'u'])
        count = 0
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                substring = s[i:j+1]
                vowels = 0
                consonants = 0
                for char in substring:
                    if char in vowels_set:
                        vowels += 1
                    else:
                        consonants += 1
                if vowels == consonants and (vowels * consonants) % k == 0:
                    count += 1
        return count