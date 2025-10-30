class Solution:
    def stringHash(self, s: str, k: int) -> str:
        n = len(s)
        # Using a list to store characters and then joining them is more efficient
        # for string building in Python than repeated string concatenation.
        result_chars = [] 

        # Iterate through the input string 's' in chunks of size 'k'.
        # The range(0, n, k) generates start indices for each substring: 0, k, 2k, ...
        for i in range(0, n, k):
            # Extract the current substring of length k
            current_substring = s[i : i + k]
            
            current_sum_of_hashes = 0
            # Calculate the sum of hash values for all characters in the current substring.
            for char_code in current_substring:
                # The hash value of a character is its 0-indexed position in the English alphabet.
                # 'a' -> 0, 'b' -> 1, ..., 'z' -> 25
                hash_value = ord(char_code) - ord('a')
                current_sum_of_hashes += hash_value
            
            # Find the remainder of the sum when divided by 26.
            # This ensures the result stays within the 0-25 range for alphabet indexing.
            hashed_char_index = current_sum_of_hashes % 26
            
            # Convert the calculated index back to its corresponding lowercase English letter.
            # For example, if hashed_char_index is 0, it becomes 'a'; if 1, it becomes 'b', etc.
            hashed_char = chr(ord('a') + hashed_char_index)
            
            # Append the newly calculated character to our list.
            result_chars.append(hashed_char)
        
        # Join all characters in the list to form the final result string.
        return "".join(result_chars)