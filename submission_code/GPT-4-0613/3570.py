class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = set('aeiou')
        n = len(word)
        count = 0
        for i in range(n):
            for j in range(i+5, n+1):
                substring = word[i:j]
                if len(set(substring).intersection(vowels)) == 5 and len([c for c in substring if c not in vowels]) == k:
                    count += 1
        return count