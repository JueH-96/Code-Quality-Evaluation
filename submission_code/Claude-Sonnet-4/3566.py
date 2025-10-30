class Solution:
    def stringSequence(self, target: str) -> List[str]:
        result = []
        current = ""
        
        for char in target:
            # First, append 'a' to the current string
            current += 'a'
            result.append(current)
            
            # Then, change the last character from 'a' to the target character
            last_char = 'a'
            while last_char != char:
                # Move to next character
                last_char = chr((ord(last_char) - ord('a') + 1) % 26 + ord('a'))
                # Update the current string
                current = current[:-1] + last_char
                result.append(current)
        
        return result