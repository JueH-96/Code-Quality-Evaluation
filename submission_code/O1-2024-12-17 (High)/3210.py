class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        # Define the set of vowel characters
        vowels = {'a', 'e', 'i', 'o', 'u'}
        
        n = len(s)
        # Precompute prefix sums for vowel counts
        prefix_vowels = [0] * (n + 1)
        for i in range(n):
            prefix_vowels[i + 1] = prefix_vowels[i] + (1 if s[i] in vowels else 0)
        
        count = 0
        # Enumerate all substrings in O(n^2)
        for start in range(n):
            for end in range(start + 1, n + 1):
                vowel_count = prefix_vowels[end] - prefix_vowels[start]
                consonant_count = (end - start) - vowel_count
                # Check the conditions for a beautiful substring
                if vowel_count == consonant_count:
                    if (vowel_count * consonant_count) % k == 0:
                        count += 1
        
        return count