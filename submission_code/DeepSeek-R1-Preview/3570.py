class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        n = len(word)
        count = 0
        
        # Precompute prefix_vowels array
        prefix_vowels = [0] * (n + 1)
        for i in range(n):
            prefix_vowels[i + 1] = prefix_vowels[i] + (1 if word[i] in vowels else 0)
        
        for i in range(n):
            vowels_seen = set()
            for j in range(i, n):
                c = word[j]
                if c in vowels:
                    vowels_seen.add(c)
                # Check if the current substring is of length >=5
                if j >= i + 4:
                    # Check if all vowels are present
                    if len(vowels_seen) == 5:
                        # Calculate consonants count
                        L = j - i + 1
                        vowels_count = prefix_vowels[j + 1] - prefix_vowels[i]
                        consonants_count = L - vowels_count
                        if consonants_count == k:
                            count += 1
        return count