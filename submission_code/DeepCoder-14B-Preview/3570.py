class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        n = len(word)
        result = 0
        
        for i in range(n):
            for j in range(i, n):
                substring = word[i:j+1]
                # Check if all vowels are present
                vowel_set = {c for c in substring if c in vowels}
                if len(vowel_set) != 5:
                    continue
                # Count consonants
                consonant_count = sum(1 for c in substring if c not in vowels)
                if consonant_count == k:
                    result += 1
        
        return result