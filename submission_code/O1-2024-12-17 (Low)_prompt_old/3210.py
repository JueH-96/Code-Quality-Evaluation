class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        # First, define a helper function to check if a character is a vowel
        def is_vowel(ch: str) -> bool:
            return ch in {'a', 'e', 'i', 'o', 'u'}
        
        n = len(s)
        
        # Precompute prefix sums: prefix_vowels[i] will hold the number
        # of vowels in s[:i], similarly for prefix_consonants
        prefix_vowels = [0] * (n + 1)
        prefix_consonants = [0] * (n + 1)
        
        for i in range(n):
            prefix_vowels[i + 1] = prefix_vowels[i] + (1 if is_vowel(s[i]) else 0)
            prefix_consonants[i + 1] = prefix_consonants[i] + (0 if is_vowel(s[i]) else 1)
        
        beautiful_count = 0
        
        # Check all substrings s[i:j+1]
        for i in range(n):
            for j in range(i, n):
                vowels_count = prefix_vowels[j + 1] - prefix_vowels[i]
                cons_count = prefix_consonants[j + 1] - prefix_consonants[i]
                
                # Check conditions:
                # 1) vowels_count == cons_count
                # 2) (vowels_count * cons_count) % k == 0
                if vowels_count == cons_count:
                    if vowels_count != 0 and (vowels_count * cons_count) % k == 0:
                        beautiful_count += 1
        
        return beautiful_count