class Solution:
    def compressedString(self, word: str) -> str:
        # If the word is empty, just return an empty string
        if not word:
            return ""
        
        # We'll build the result in a list for efficiency
        parts = []
        prev_char = word[0]
        count = 1
        
        # Iterate over the string starting from the second character
        for ch in word[1:]:
            # If it's the same character and we haven't reached 9 repeats yet
            if ch == prev_char and count < 9:
                count += 1
            else:
                # Flush the current run
                parts.append(f"{count}{prev_char}")
                # Start a new run
                prev_char = ch
                count = 1
        
        # Flush the final run
        parts.append(f"{count}{prev_char}")
        
        # Join all parts and return
        return "".join(parts)