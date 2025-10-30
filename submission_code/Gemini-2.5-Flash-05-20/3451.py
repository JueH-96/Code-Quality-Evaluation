class Solution:
    def compressedString(self, word: str) -> str:
        # Use a list to build the compressed string parts efficiently.
        # Appending to a list and then joining once at the end is generally faster
        # than repeatedly concatenating strings in Python.
        comp_parts = []
        n = len(word)
        
        # 'i' is the main pointer, indicating the current starting position
        # in the 'word' string that we are processing.
        i = 0
        
        # Continue processing until the entire 'word' has been traversed.
        while i < n:
            # Get the character that starts the current segment.
            char = word[i]
            
            # 'count' will store the length of the current prefix of 'char'.
            # This count must not exceed 9 as per the problem description.
            count = 0
            
            # 'j' is a look-ahead pointer. It helps in finding how many
            # consecutive occurrences of 'char' are present starting from 'i',
            # limited to a maximum of 9 characters.
            j = i 
            
            # Iterate as long as:
            # 1. 'j' is within the bounds of the 'word' string.
            # 2. The character at 'word[j]' is the same as 'char'.
            # 3. The 'count' of consecutive characters is less than 9.
            while j < n and word[j] == char and count < 9:
                count += 1  # Increment the count for the current character run.
                j += 1      # Move the look-ahead pointer to the next character.
            
            # After the inner loop, 'count' holds the length of the segment
            # (at most 9) and 'char' is the character itself.
            # Append the count (converted to a string) and the character to our list.
            comp_parts.append(str(count))
            comp_parts.append(char)
            
            # Move the main pointer 'i' to the position immediately after the
            # segment that was just processed. This ensures we start the next
            # iteration from the correct place in 'word'.
            i = j
            
        # Join all the collected parts from the list into a single string
        # and return the final compressed result.
        return "".join(comp_parts)