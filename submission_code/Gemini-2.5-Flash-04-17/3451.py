class Solution:
    def compressedString(self, word: str) -> str:
        n = len(word)
        # The constraint 1 <= word.length means n will always be at least 1.
        # No need for the empty string check 'if n == 0:'.

        comp = [] # Using a list to build the result is more efficient
        i = 0     # Pointer for the current position in the word

        while i < n:
            char = word[i] # Character to count
            count = 0      # Counter for consecutive occurrences
            j = i          # Inner pointer to scan for consecutive chars

            # Count consecutive occurrences of 'char' starting from index 'i'
            # The count is limited to a maximum of 9
            # We stop if we reach the end of the string (j < n)
            # or if the character changes (word[j] == char)
            # or if we have counted 9 characters already (count < 9)
            while j < n and word[j] == char and count < 9:
                count += 1
                j += 1 # Move the inner pointer

            # Append the count (as a string) and the character to the result list
            comp.append(str(count))
            comp.append(char)

            # Move the main pointer 'i' forward by the number of characters just processed
            i += count

        # Join the list elements into a single string and return it
        return "".join(comp)