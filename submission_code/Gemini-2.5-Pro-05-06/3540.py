class Solution:
    def stringHash(self, s: str, k: int) -> str:
        n = len(s)
        
        # Using a list to store characters of the result string.
        # Appending to a list and then joining is generally more efficient
        # in Python for building strings in a loop compared to repeated string concatenation.
        result_chars_list = []
        
        # Iterate over the string s in chunks of size k.
        # The `range` function's third argument is the step.
        # This loop will set `i` to 0, k, 2k, ..., up to the last multiple of k strictly less than n.
        # Since n is a multiple of k, the last `i` will be `n - k`.
        for i in range(0, n, k):
            # Extract the current substring. In Python, s[start:end] creates a new string (a copy).
            # The substring will be s[i], s[i+1], ..., s[i+k-1].
            substring = s[i : i + k]
            
            sum_of_hash_values = 0
            # For each character in the current substring:
            for char_in_substring in substring:
                # Calculate its hash value.
                # As per the problem: 'a' corresponds to 0, 'b' to 1, ..., 'z' to 25.
                # `ord(char)` gives its ASCII value. `ord('a')` is the ASCII value of 'a'.
                # Their difference `ord(char_in_substring) - ord('a')` gives the 0-based index.
                hash_value = ord(char_in_substring) - ord('a')
                sum_of_hash_values += hash_value
            
            # Find the remainder of this sum when divided by 26.
            # The problem statement refers to this numeric value as "hashedChar".
            # Python's `%` operator computes the remainder. For non-negative `sum_of_hash_values`,
            # the result will be in the range [0, 25], which is what we need.
            hashed_char_code = sum_of_hash_values % 26
            
            # Convert this numeric code (0-25) back to its corresponding lowercase English character.
            # (e.g., 0 maps to 'a', 1 to 'b', ..., 25 to 'z').
            # `chr()` converts an ASCII (or Unicode code point) value back to a character.
            # `ord('a') + hashed_char_code` gives the ASCII value of the target character.
            final_char_for_substring = chr(ord('a') + hashed_char_code)
            
            # Append the resulting character to our list.
            result_chars_list.append(final_char_for_substring)
            
        # Join all characters in the list to form the final result string.
        return "".join(result_chars_list)