class Solution:
    def stringSequence(self, target: str) -> List[str]:
        result = []
        current_string = ""
        
        for char in target:
            # Append 'a' until we reach the desired character
            if not current_string:
                current_string = 'a'
                result.append(current_string)
            
            while current_string[-1] != char:
                if current_string[-1] == 'z':
                    current_string = current_string[:-1] + 'a'
                else:
                    current_string = current_string[:-1] + chr(ord(current_string[-1]) + 1)
                result.append(current_string)
            
            # Append 'a' for the next character in the target
            current_string += 'a'
            result.append(current_string)
        
        # Remove the last 'a' appended as it is not needed
        result.pop()
        
        return result