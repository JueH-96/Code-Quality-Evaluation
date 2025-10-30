class Solution:
    def compressedString(self, word: str) -> str:
        # Initialize the result list for efficiency
        parts = []
        n = len(word)
        if n == 0:
            return ""
        
        # Start with the first character
        current_char = word[0]
        count = 1
        
        # Iterate over the string starting from the second character
        for i in range(1, n):
            if word[i] == current_char and count < 9:
                # Continue the current run if same char and count < 9
                count += 1
            else:
                # Flush the current run
                parts.append(str(count))
                parts.append(current_char)
                # Start a new run
                current_char = word[i]
                count = 1
        
        # Flush the final run
        parts.append(str(count))
        parts.append(current_char)
        
        # Join all parts into the final compressed string
        return "".join(parts)