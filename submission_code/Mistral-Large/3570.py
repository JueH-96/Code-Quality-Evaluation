class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = set('aeiou')
        n = len(word)
        count = 0

        # Helper function to check if a substring contains all vowels
        def has_all_vowels(substr):
            return vowels.issubset(set(substr))

        # Helper function to count consonants in a substring
        def count_consonants(substr):
            return sum(1 for char in substr if char not in vowels)

        # Generate all possible substrings
        for i in range(n):
            for j in range(i + 1, n + 1):
                substr = word[i:j]
                if has_all_vowels(substr) and count_consonants(substr) == k:
                    count += 1

        return count