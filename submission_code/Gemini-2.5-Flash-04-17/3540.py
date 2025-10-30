class Solution:
    def stringHash(self, s: str, k: int) -> str:
        n = len(s)
        
        # The result string will have length n / k
        # We process the input string s in chunks of size k

        result_chars = [] # Use a list to efficiently build the result string

        # Iterate through the string s using a step of k
        # 'i' will be the starting index of each k-sized block
        # The range function with step k generates starting indices: 0, k, 2*k, ..., (n/k - 1)*k
        for i in range(0, n, k):
            # 'i' represents the start index of the current k-sized block
            
            current_sum = 0
            # Calculate the sum of hash values for characters in the current block
            # The characters for the block starting at index 'i' are s[i], s[i+1], ..., s[i+k-1]
            for j in range(k):
                # Get the character at the current index within the block
                char = s[i + j]
                
                # Calculate the hash value of the character
                # The hash value is the 0-based index in the English alphabet ('a' -> 0, 'b' -> 1, ...)
                char_hash = ord(char) - ord('a')
                
                # Add the hash value to the current sum for this block
                current_sum += char_hash
            
            # Calculate the final hash value for this block
            # This is the remainder when the sum of character hash values is divided by 26
            hashed_char_index = current_sum % 26
            
            # Convert the calculated index back to a lowercase English character
            # 0 corresponds to 'a', 1 to 'b', ..., 25 to 'z'
            hashed_char = chr(ord('a') + hashed_char_index)
            
            # Append the resulting hashed character for this block to our list
            result_chars.append(hashed_char)
        
        # Join the list of characters into a single string to form the final result
        return "".join(result_chars)