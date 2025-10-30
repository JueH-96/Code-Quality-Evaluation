class Solution:
    def kthCharacter(self, k: int) -> str:
        # Start with the initial word as a list of characters for efficient modification/extension.
        word_chars = ['a']
        
        # Helper function to get the next character in the English alphabet, wrapping 'z' to 'a'.
        # This uses modular arithmetic on ASCII values.
        def next_char(c):
            # ord(c) gets the integer ASCII value of character c.
            # ord(c) - ord('a') gives the 0-indexed position of the character in the alphabet (0 for 'a', 1 for 'b', ..., 25 for 'z').
            # Add 1 to get the position of the next character.
            # The modulo 26 (%) operation handles the wrap-around: if the position is 26 (after 'z'), it becomes 0 ('a').
            # Add ord('a') back to convert the 0-indexed position back to the ASCII value of the character.
            # chr() converts the ASCII value back to a character.
            return chr((ord(c) - ord('a') + 1) % 26 + ord('a'))

        # Simulate the word generation process according to the rule.
        # We continue performing the operation as long as the current word length is less than k.
        # The length of the list `word_chars` represents the current length of the word.
        while len(word_chars) < k:
            generated_chars = []
            # Iterate through each character in the current word (represented by word_chars).
            # For each character, find its next character and add it to a temporary list.
            for char in word_chars:
                generated_chars.append(next_char(char))
            # Append the sequence of generated characters to the end of the current word_chars list.
            # This effectively doubles the length of the word, just like appending a string.
            word_chars.extend(generated_chars)

        # After the loop finishes, the length of word_chars is guaranteed to be at least k.
        # The problem asks for the k-th character, which is 1-indexed.
        # In Python (and most programming languages), sequences are 0-indexed.
        # The k-th element is located at index k-1.
        return word_chars[k-1]