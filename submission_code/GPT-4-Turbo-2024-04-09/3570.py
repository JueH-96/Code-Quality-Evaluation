class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        from collections import Counter
        
        def has_all_vowels_and_k_consonants(counter, k):
            vowels = {'a', 'e', 'i', 'o', 'u'}
            if all(counter[v] > 0 for v in vowels):
                consonant_count = sum(count for char, count in counter.items() if char not in vowels)
                return consonant_count == k
            return False
        
        n = len(word)
        result = 0
        
        for start in range(n):
            char_count = Counter()
            consonant_count = 0
            for end in range(start, n):
                char = word[end]
                char_count[char] += 1
                if char not in 'aeiou':
                    consonant_count += 1
                if consonant_count > k:
                    break
                if has_all_vowels_and_k_consonants(char_count, k):
                    result += 1
        
        return result