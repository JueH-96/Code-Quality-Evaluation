class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = 'aeiouAEIOU'
        vowel_positions = []
        vowel_chars = []

        # Collect vowels and their positions
        for i, char in enumerate(s):
            if char in vowels:
                vowel_positions.append(i)
                vowel_chars.append(char)

        # Sort the vowels
        vowel_chars.sort()

        # Create a list from the original string to modify it
        result = list(s)

        # Place sorted vowels back into their original positions
        for pos, char in zip(vowel_positions, vowel_chars):
            result[pos] = char

        return ''.join(result)