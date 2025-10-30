class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        n = len(word)
        consonant_counts = [0] * (n + 1)
        for i in range(n):
            consonant_counts[i+1] = consonant_counts[i] + (1 if word[i] not in vowels else 0)
        count = 0
        for i in range(n):
            for j in range(i+1, n+1):
                substr_len = j - i
                if substr_len < 5:
                    continue
                # Check if all vowels are present in the substring
                if all(c in word[i:j] for c in vowels):
                    # Calculate the number of consonants
                    consonants = consonant_counts[j] - consonant_counts[i]
                    if consonants == k:
                        count += 1
        return count