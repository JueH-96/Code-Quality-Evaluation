class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        count = 0
        vowels_set = {'a', 'e', 'i', 'o', 'u'}

        # Iterate through all possible starting indices of substrings
        for i in range(n):
            current_vowels = 0
            current_consonants = 0
            
            # Iterate through all possible ending indices, extending the substring
            # from `s[i]` to `s[j]`
            for j in range(i, n):
                char = s[j]
                
                # Update vowel/consonant counts for the current substring s[i...j]
                if char in vowels_set:
                    current_vowels += 1
                else:
                    current_consonants += 1
                
                # Check the beauty conditions for the current substring
                # Condition 1: Number of vowels must equal number of consonants
                if current_vowels == current_consonants:
                    # Condition 2: (vowels * consonants) must be divisible by k
                    # Since current_vowels == current_consonants, this simplifies to
                    # (current_vowels * current_vowels) % k == 0
                    if (current_vowels * current_consonants) % k == 0:
                        count += 1
        
        return count