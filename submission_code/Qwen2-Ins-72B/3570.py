class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = set('aeiou')
        n = len(word)
        ans = 0
        for i in range(n):
            if word[i] not in vowels:  # Skip if the character is not a vowel
                continue
            vowel_set = set()
            consonant_count = 0
            for j in range(i, n):
                if word[j] in vowels:
                    vowel_set.add(word[j])
                else:
                    consonant_count += 1
                if len(vowel_set) == 5 and consonant_count == k:
                    ans += 1
        return ans