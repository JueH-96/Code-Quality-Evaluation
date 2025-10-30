class Solution:
    def stringSequence(self, target: str) -> List[str]:
        result = []
        current = ""
        
        for char in target:
            # First append 'a'
            current += 'a'
            result.append(current)
            
            # Then transform 'a' to the target character
            while current[-1] != char:
                # Change last character to next in alphabet
                last_char = current[-1]
                next_char = chr((ord(last_char) - ord('a') + 1) % 26 + ord('a'))
                current = current[:-1] + next_char
                result.append(current)
        
        return result