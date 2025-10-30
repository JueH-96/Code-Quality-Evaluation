class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = set('aeiou')
        n = len(word)
        count = 0
        for i in range(n):
            for j in range(i, n):
                sub = word[i:j+1]
                vowel_set = set()
                consonant_count = 0
                for char in sub:
                    if char in vowels:
                        vowel_set.add(char)
                    else:
                        consonant_count += 1
                if len(vowel_set) == 5 and consonant_count == k:
                    count += 1
        return count