class Solution:
    def compressedString(self, word: str) -> str:
        """
        Compresses a string according to the specified algorithm.

        The algorithm iterates through the string, identifying consecutive runs of
        the same character. Each run is then broken down into chunks of at most 9
        characters. For each chunk, its length and the character are appended to
        the result.
        """
        comp = []
        i = 0
        n = len(word)
        
        while i < n:
            current_char = word[i]
            count = 0
            
            # Use a temporary pointer 'j' to scan ahead for the current segment.
            # A segment is a run of the same character, with a maximum length of 9.
            j = i
            while j < n and word[j] == current_char and count < 9:
                count += 1
                j += 1
            
            # Append the compressed form of the segment (count + character).
            comp.append(str(count))
            comp.append(current_char)
            
            # Move the main pointer 'i' to the start of the next segment.
            i = j
            
        return "".join(comp)