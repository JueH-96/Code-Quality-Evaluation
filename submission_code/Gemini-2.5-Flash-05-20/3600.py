class Solution:
    def kthCharacter(self, k: int) -> str:
        word = "a"

        # Helper function to get the next character in the English alphabet.
        # 'z' wraps around to 'a'.
        def get_next_char(char_val: str) -> str:
            # Convert the character to its 0-indexed position (a=0, b=1, ..., z=25).
            char_code_relative_to_a = ord(char_val) - ord('a')
            
            # Calculate the next character's 0-indexed position.
            # Add 1 to move to the next letter. The modulo 26 handles the wrap-around (z+1 becomes a).
            next_char_code_relative_to_a = (char_code_relative_to_a + 1) % 26
            
            # Convert the 0-indexed position back to its corresponding ASCII character.
            return chr(next_char_code_relative_to_a + ord('a'))

        # The loop continues until the 'word' string is long enough to contain the k-th character.
        # Since the length of 'word' doubles in each iteration, this loop will run a maximum
        # of log2(k) times (e.g., for k=500, max ~9 iterations).
        while len(word) < k:
            # We build the 'next_word_part' as a list of characters first.
            # This is generally more efficient in Python than repeated string concatenations,
            # especially for potentially larger strings (though here max length is 512).
            next_word_part_chars = []
            for char_in_word in word:
                next_word_part_chars.append(get_next_char(char_in_word))
            
            # Join the list of characters to form the string that will be appended.
            next_word_part = "".join(next_word_part_chars)
            
            # Append the newly generated part to the existing word.
            word += next_word_part
        
        # After the loop, 'word' has at least k characters.
        # Since k is 1-indexed (e.g., 1st character), we access word[k-1] for 0-indexed string access.
        return word[k-1]