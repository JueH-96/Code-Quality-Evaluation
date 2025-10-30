class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        vowels_set = {'a', 'e', 'i', 'o', 'u'}
        n = len(s)
        
        # Prefix sums for counting vowels and consonants
        prefix_vowels = [0] * (n + 1)
        prefix_cons = [0] * (n + 1)
        
        for i in range(n):
            prefix_vowels[i + 1] = prefix_vowels[i] + (s[i] in vowels_set)
            prefix_cons[i + 1] = prefix_cons[i] + (s[i] not in vowels_set)
        
        beautiful_count = 0
        
        # Check all substrings in O(n^2)
        for start in range(n):
            for end in range(start, n):
                # Number of vowels and consonants in substring s[start:end+1]
                v_count = prefix_vowels[end + 1] - prefix_vowels[start]
                c_count = prefix_cons[end + 1] - prefix_cons[start]
                
                # Check the two conditions for being beautiful
                if v_count == c_count:
                    if (v_count * c_count) % k == 0:
                        beautiful_count += 1
        
        return beautiful_count