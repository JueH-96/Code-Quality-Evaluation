import math # Not strictly needed, but good habit if math functions were used.

class Solution:
    def stringHash(self, s: str, k: int) -> str:
        """
        Hashes a string s into a new string result of length n / k.

        Args:
            s: The input string consisting of lowercase English letters.
            k: The length of substrings to process. s.length is divisible by k.

        Returns:
            The hashed string result.
        """
        n = len(s)
        
        # Ensure n is divisible by k as per constraints, although the problem statement guarantees this.
        # if n % k != 0:
        #     raise ValueError("Length of s must be divisible by k")
            
        result_chars = [] # Use a list to store characters for efficient appending

        # Iterate through the string s in steps of k
        # The loop goes from index 0 up to n-k (inclusive), stepping by k.
        # range(0, n, k) generates indices 0, k, 2k, ..., up to the last multiple of k less than n.
        for i in range(0, n, k):
            # Extract the substring of length k starting at index i
            substring = s[i : i + k]
            
            current_sum = 0
            # Calculate the sum of hash values for the characters in the substring
            for char in substring:
                # The hash value of a character is its 0-based index in the alphabet
                hash_value = ord(char) - ord('a') 
                current_sum += hash_value
                
            # Find the remainder when the sum is divided by 26
            hashed_char_value = current_sum % 26
            
            # Convert the remainder (0-25) back to the corresponding lowercase character
            # ord('a') gives the ASCII value of 'a'. Adding the hashed_char_value gives
            # the ASCII value of the target character. chr() converts ASCII back to char.
            result_char = chr(ord('a') + hashed_char_value)
            
            # Append the resulting character to our list
            result_chars.append(result_char)
            
        # Join the characters in the list to form the final result string
        return "".join(result_chars)