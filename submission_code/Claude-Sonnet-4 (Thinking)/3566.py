class Solution:
    def stringSequence(self, target: str) -> List[str]:
        result = []
        current = ""
        
        for i in range(len(target)):
            # First, we need to have at least i+1 characters
            # So if current length is i, we need to append "a"
            if len(current) == i:
                current += "a"
                result.append(current)
            
            # Now we need to transform the last character to target[i]
            while current[-1] != target[i]:
                # Transform last character to next character
                last_char = current[-1]
                if last_char == 'z':
                    next_char = 'a'
                else:
                    next_char = chr(ord(last_char) + 1)
                current = current[:-1] + next_char
                result.append(current)
        
        return result