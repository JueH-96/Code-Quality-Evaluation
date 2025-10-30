class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        vowels_set = {'a', 'e', 'i', 'o', 'u'}
        N = len(s)
        
        # Prefix sums for vowels and consonants
        vowels_count = [0] * (N + 1)
        consonants_count = [0] * (N + 1)
        
        for i in range(N):
            vowels_count[i+1] = vowels_count[i] + (1 if s[i] in vowels_set else 0)
            consonants_count[i+1] = consonants_count[i] + (1 if s[i] not in vowels_set else 0)
        
        count = 0
        for i in range(N):
            for j in range(i, N):
                v = vowels_count[j+1] - vowels_count[i]
                c = consonants_count[j+1] - consonants_count[i]
                if v == c and (v * c) % k == 0:
                    count += 1
        return count