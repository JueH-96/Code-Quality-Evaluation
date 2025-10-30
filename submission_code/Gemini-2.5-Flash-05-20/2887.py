class Solution:
    def sortVowels(self, s: str) -> str:
        # Define a set of all possible vowels (lowercase and uppercase)
        VOWELS = {'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'}

        # Helper function to check if a character is a vowel
        def is_vowel(char: str) -> bool:
            return char in VOWELS

        # Step 1: Collect all vowels from the input string s
        vowels_in_s = []
        for char in s:
            if is_vowel(char):
                vowels_in_s.append(char)

        # Step 2: Sort the collected vowels by their ASCII values
        # Python's default sort for strings sorts by ASCII value
        vowels_in_s.sort()

        # Step 3: Construct the result string
        # Convert the input string s to a list of characters
        # This allows for modification of characters by index
        result_chars = list(s)
        
        # Initialize a pointer for the sorted vowels list
        # This will tell us which sorted vowel to place next
        vowel_idx = 0

        # Iterate through the original string's indices
        for i in range(len(s)):
            # If the character at the current index in the original string is a vowel,
            # replace it with the next sorted vowel from our collected list.
            if is_vowel(s[i]):
                result_chars[i] = vowels_in_s[vowel_idx]
                vowel_idx += 1 # Move to the next sorted vowel
        
        # Join the list of characters back into a string to get the final result
        return "".join(result_chars)