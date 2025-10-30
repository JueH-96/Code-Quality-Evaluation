class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        vowels = set('aeiou')
        count = 0
        
        for i in range(len(s)):
            for j in range(i, len(s)+1):
                substring = s[i:j]
                if len(substring) > 0:
                    vowel_count = sum(1 for char in substring if char in vowels)
                    consonant_count = len(substring) - vowel_count
                    if vowel_count == consonant_count and (vowel_count * consonant_count) % k == 0:
                        count += 1
        
        return count