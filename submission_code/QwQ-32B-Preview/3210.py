class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        vowels = set('aeiou')
        n = len(s)
        
        # Precompute cumulative counts of vowels and consonants
        vowels_count = [0] * (n + 1)
        consonants_count = [0] * (n + 1)
        for i in range(n):
            if s[i] in vowels:
                vowels_count[i + 1] = vowels_count[i] + 1
                consonants_count[i + 1] = consonants_count[i]
            else:
                consonants_count[i + 1] = consonants_count[i] + 1
                vowels_count[i + 1] = vowels_count[i]
        
        count = 0
        # Iterate over all possible substrings
        for l in range(n):
            for r in range(l + 1, n + 1):
                v = vowels_count[r] - vowels_count[l]
                c = consonants_count[r] - consonants_count[l]
                if v == c and (v * c) % k == 0:
                    count += 1
        return count