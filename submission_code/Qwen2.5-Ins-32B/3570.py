class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = set(['a', 'e', 'i', 'o', 'u'])
        n = len(word)
        count = 0
        
        def has_all_vowels(s):
            return vowels.issubset(set(s))
        
        for i in range(n):
            for j in range(i + 5, n + 1):
                substring = word[i:j]
                consonants = sum(1 for c in substring if c not in vowels)
                if consonants == k and has_all_vowels(substring):
                    count += 1
                    
        return count