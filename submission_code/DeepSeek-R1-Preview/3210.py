class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        prefix = [0] * (n + 1)
        vowels_set = {'a', 'e', 'i', 'o', 'u'}
        
        for i in range(n):
            if s[i] in vowels_set:
                prefix[i + 1] = prefix[i] + 1
            else:
                prefix[i + 1] = prefix[i]
        
        count = 0
        for i in range(n):
            for j in range(i, n):
                length = j - i + 1
                if length % 2 != 0:
                    continue
                vowels = prefix[j + 1] - prefix[i]
                consonants = length - vowels
                if vowels == consonants:
                    product = vowels * consonants
                    if product % k == 0:
                        count += 1
        return count