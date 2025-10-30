class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = set('aeiou')
        n = len(word)
        total_count = 0

        # Helper function to check if a substring contains all vowels
        def contains_all_vowels(substring):
            return all(v in substring for v in vowels)

        # Iterate over all possible substrings
        for start in range(n):
            vowel_count = 0
            consonant_count = 0
            seen_vowels = set()
            
            for end in range(start, n):
                char = word[end]
                
                if char in vowels:
                    seen_vowels.add(char)
                else:
                    consonant_count += 1
                
                # If we have more consonants than k, break early
                if consonant_count > k:
                    break
                
                # Check if we have all vowels and exactly k consonants
                if len(seen_vowels) == 5 and consonant_count == k:
                    total_count += 1

        return total_count