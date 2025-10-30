class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        vowels_set = {'a', 'e', 'i', 'o', 'u'}
        n = len(s)
        
        # Prefix sums for vowels and consonants
        prefix_vowels = [0] * (n + 1)
        prefix_cons = [0] * (n + 1)
        
        for i in range(n):
            prefix_vowels[i + 1] = prefix_vowels[i] + (1 if s[i] in vowels_set else 0)
            prefix_cons[i + 1] = prefix_cons[i] + (0 if s[i] in vowels_set else 1)
        
        beautiful_count = 0
        
        # Check all substrings using prefix sums
        for start in range(n):
            for end in range(start, n):
                v = prefix_vowels[end + 1] - prefix_vowels[start]
                c = prefix_cons[end + 1] - prefix_cons[start]
                
                # Check conditions for being beautiful
                if v == c and (v * c) % k == 0:
                    beautiful_count += 1
        
        return beautiful_count