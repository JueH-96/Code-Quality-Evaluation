class Solution:
    def stringHash(self, s: str, k: int) -> str:
        # Get the length of the input string.
        n = len(s)
        
        # Initialize an empty list to store the characters of the result string.
        # Using a list and then joining is more efficient than repeated string concatenation.
        result_chars = []
        
        # The problem states that n is a multiple of k.
        # We iterate through the string in non-overlapping chunks of size k.
        # The `range` function with a step is perfect for this.
        # `i` will be the starting index of each chunk: 0, k, 2k, ...
        for i in range(0, n, k):
            # Extract the substring of length k for the current chunk.
            substring = s[i : i + k]
            
            # Calculate the sum of the hash values of the characters in the substring.
            # The hash value of a character is its 0-based index in the alphabet.
            # We use a generator expression within the `sum()` function for a concise calculation.
            # `ord(char) - ord('a')` computes the hash value (e.g., 'a' -> 0, 'b' -> 1).
            hash_sum = sum(ord(char) - ord('a') for char in substring)
            
            # Find the remainder of this sum when divided by 26. This is the `hashedChar` value.
            hashed_char_value = hash_sum % 26
            
            # Convert the numeric `hashedChar` value back into a lowercase English character.
            # `ord('a')` gives the ASCII value of 'a'. We add our 0-25 value to it
            # and then convert the resulting ASCII code back to a character with `chr()`.
            final_char = chr(ord('a') + hashed_char_value)
            
            # Append the newly computed character to our list.
            result_chars.append(final_char)
            
        # Join all the characters in the list to form the final result string and return it.
        return "".join(result_chars)