class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        n = len(word)
        # Create prefix sum arrays for each vowel
        prefix = {v: [0] * (n + 1) for v in vowels}
        for i in range(n):
            current_char = word[i]
            # Copy previous counts
            for v in vowels:
                prefix[v][i+1] = prefix[v][i]
            # Update the count for the current character if it's a vowel
            if current_char in vowels:
                prefix[current_char][i+1] += 1
        
        result = 0
        # Iterate over all possible substrings
        for i in range(n):
            for j in range(i, n):
                # Check if all vowels are present and count consonants
                valid = True
                sum_vowels = 0
                for v in vowels:
                    count = prefix[v][j+1] - prefix[v][i]
                    sum_vowels += count
                    if count == 0:
                        valid = False
                length = j - i + 1
                consonants = length - sum_vowels
                if valid and consonants == k:
                    result += 1
        return result