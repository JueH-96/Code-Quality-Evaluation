class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        vowels = {'a', 'e', 'i', 'o', 'u'}
        total = 0
        
        # Try every possible starting index
        for i in range(n):
            seen_vowels = set()
            consonant_count = 0
            
            # Extend the substring from i to j
            for j in range(i, n):
                ch = word[j]
                
                # Update vowel or consonant count
                if ch in vowels:
                    seen_vowels.add(ch)
                else:
                    consonant_count += 1
                    # If we exceed k consonants, no longer valid and no need to go further
                    if consonant_count > k:
                        break
                
                # If we've seen all 5 vowels and used exactly k consonants, count it
                if len(seen_vowels) == 5 and consonant_count == k:
                    total += 1
        
        return total