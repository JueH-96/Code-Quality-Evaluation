import sys 
# sys.setrecursionlimit(2000) # Setting higher recursion depth is generally not needed for iterative solutions like this one.

class Solution:
    """
    Implements the compression algorithm described in the problem statement.
    The algorithm iterates through the input string 'word' and compresses consecutive
    identical characters into chunks of length at most 9. Each chunk is represented
    by its length followed by the character.
    """
    def compressedString(self, word: str) -> str:
        """
        Compresses the input string 'word' according to the specified algorithm:
        Iteratively remove a maximum length prefix of 'word' made of a single character 'c'
        repeating at most 9 times. Append the length of the prefix followed by 'c' to the result string.

        Args:
            word: The input string consisting of lowercase English letters. 
                  Constraints: 1 <= word.length <= 2 * 10^5.

        Returns:
            The compressed string. For example, "aaaaaaaaaaaaaabb" becomes "9a5a2b".
        """
        n = len(word)
        # Constraints state 1 <= word.length, so we don't need to handle the empty string case.
        
        comp_parts = [] # Use a list to store parts of the compressed string for efficient concatenation
        i = 0 # Initialize a pointer 'i' to traverse the input string 'word'

        # Iterate through the word using the pointer 'i'
        while i < n:
            current_char = word[i] # Get the character at the current starting position of a potential chunk
            count = 0 # Initialize the count for the current chunk of identical characters
            
            # Use a second pointer 'j' to find the end of the consecutive sequence of 'current_char'
            j = i 
            
            # Inner loop conditions determine the chunk:
            # 1. j < n: Ensure we are within the string bounds.
            # 2. word[j] == current_char: Check if the character is the same as the start of the chunk.
            # 3. count < 9: Limit the maximum length of the chunk to 9 as per the algorithm requirement.
            while j < n and word[j] == current_char and count < 9:
                count += 1 # Increment the count for this chunk
                j += 1 # Move the inner pointer 'j' forward to check the next character
            
            # After the inner loop, 'count' holds the length of the identified chunk (from 1 to 9).
            # Append the length (converted to a string) and the character itself to the list of parts.
            comp_parts.append(str(count))
            comp_parts.append(current_char)
            
            # Move the main pointer 'i' forward by 'count' positions.
            # This correctly positions 'i' at the beginning of the next potential chunk or at the end of the string.
            i += count 

        # Join all the collected parts (lengths and characters) into a single final string.
        # Using "".join() is an efficient method for string concatenation from list elements in Python.
        comp = "".join(comp_parts)
        
        return comp