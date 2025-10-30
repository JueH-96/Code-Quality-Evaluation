class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        # Collect all vowels in the string
        vowels_list = [c for c in s if c.lower() in vowels]
        # Sort the vowels based on their ASCII values
        vowels_sorted = sorted(vowels_list)
        # Use an iterator to efficiently get the next sorted vowel
        vowel_iter = iter(vowels_sorted)
        # Build the result string
        result = []
        for char in s:
            if char.lower() in vowels:
                result.append(next(vowel_iter))
            else:
                result.append(char)
        return ''.join(result)