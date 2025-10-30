class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        result = 0
        n = len(word)
        for i in range(n):
            vowel_counts = {'a': 0, 'e': 0, 'i': 0, 'o': 0, 'u': 0}
            consonant_count = 0
            for j in range(i, n):
                char = word[j]
                if char in vowels:
                    vowel_counts[char] += 1
                else:
                    consonant_count += 1
                if all(count >= 1 for count in vowel_counts.values()) and consonant_count == k:
                    result += 1
        return result