class Solution:
    def possibleStringCount(self, word: str) -> int:
        # Step 1: Perform Run-Length Encoding (RLE) on the input word.
        # This transforms a string like "abbcccc" into a list of tuples:
        # [('a', 1), ('b', 2), ('c', 4)]
        
        rle_blocks = []
        if not word:
            # According to constraints, word.length is at least 1,
            # so this case for an empty string won't be hit.
            return 0 
        
        i = 0
        while i < len(word):
            char = word[i]
            count = 0
            j = i
            # Count consecutive occurrences of the current character
            while j < len(word) and word[j] == char:
                count += 1
                j += 1
            rle_blocks.append((char, count))
            i = j # Move pointer to the start of the next block
        
        # Initialize the total count of possible original strings.
        total_possible_strings = 0

        # Scenario 1: Alice did not press any key for too long.
        # In this case, the original string is exactly the same as the given 'word'.
        # This is always one possible original string.
        total_possible_strings += 1

        # Scenario 2: Alice pressed exactly one key for too long.
        # This means that there is exactly one contiguous block of identical characters
        # in the final 'word' that originated from a shorter sequence of the same character
        # in the original string.
        
        # Iterate through each (character, length) block obtained from the RLE.
        for char, length in rle_blocks:
            # If a block's length is greater than 1, it could have been the result
            # of a character being pressed for too long.
            if length > 1:
                # If this specific block (e.g., 'cccc' with length 4) was the one
                # where the key was pressed too long, its corresponding original
                # length could have been any value from 1 up to (length - 1).
                # For 'cccc' (length 4), the original could have been 'c' (length 1),
                # 'cc' (length 2), or 'ccc' (length 3).
                # Each of these (length - 1) possible original lengths creates a unique
                # original string where only this specific block's length was altered.
                total_possible_strings += (length - 1)
        
        return total_possible_strings