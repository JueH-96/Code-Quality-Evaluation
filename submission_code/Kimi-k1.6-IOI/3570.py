class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        n = len(word)
        result = 0
        
        for i in range(n):
            current_vowels = set()
            consonant_count = 0
            for j in range(i, n):
                char = word[j]
                if char in vowels:
                    current_vowels.add(char)
                else:
                    consonant_count += 1
                if len(current_vowels) == 5 and consonant_count == k:
                    result += 1
        
        return result