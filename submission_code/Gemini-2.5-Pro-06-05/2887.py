class Solution:
    def sortVowels(self, s: str) -> str:
        # A set of all vowels, both lowercase and uppercase, for efficient O(1) checking.
        vowels_set = frozenset('aeiouAEIOU')

        # Step 1: Extract all vowels from the input string `s` and sort them
        # in nondecreasing order of their ASCII values.
        sorted_vowels = sorted([char for char in s if char in vowels_set])

        # Step 2: Construct the new string by iterating through the original string.
        # Consonants are kept in their original positions.
        # Vowel positions are filled with vowels from the sorted list, in order.

        # Create an iterator for the sorted vowels. This provides them one by one.
        vowel_iter = iter(sorted_vowels)

        # Use a list comprehension to build the list of characters for the new string.
        # For each character in the original string `s`:
        # - If it's a vowel, take the next one from our sorted vowel iterator.
        # - If it's a consonant, keep the original character.
        result_chars = [next(vowel_iter) if char in vowels_set else char for char in s]

        # Step 3: Join the characters to form the final result string.
        return "".join(result_chars)