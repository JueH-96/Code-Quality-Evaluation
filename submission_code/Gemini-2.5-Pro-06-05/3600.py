class Solution:
    def kthCharacter(self, k: int) -> str:
        """
        Calculates the k-th character in a recursively generated string.

        The generation process creates a string where the character at any 0-indexed
        position 'i' is determined by the number of set bits (1s) in the binary
        representation of 'i'. This is known as the population count (popcount).
        """
        
        # The problem uses 1-based indexing for k. We convert it to a 0-based index.
        index = k - 1
        
        # Calculate the popcount of the index.
        # bin(index) returns a string like '0b101', so we count the '1's.
        popcount = bin(index).count('1')
        
        # The final character is 'a' shifted by the popcount value.
        # The alphabet has 26 letters, so the character value wraps around.
        # The offset from 'a' is the popcount modulo 26.
        offset = popcount % 26
        
        # Calculate the final character by adding the offset to the ASCII value of 'a'.
        return chr(ord('a') + offset)