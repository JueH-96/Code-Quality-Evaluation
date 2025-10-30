class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = set('aeiou')
        count = 0
        
        for i in range(len(word)):
            for j in range(i + 5, len(word) + 1):
                substring = word[i:j]
                if set(substring).issuperset(vowels) and sum(1 for char in substring if char not in vowels) == k:
                    count += 1
                    
        return count