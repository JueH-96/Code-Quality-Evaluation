class Solution:
    def stringSequence(self, target: str) -> List[str]:
        result = []
        current_string = ""
        
        for char in target:
            # Append 'a' until the length of current_string is one more than the previous
            current_string += 'a'
            result.append(current_string)
            
            # Change the last character to match the target character
            while current_string[-1] != char:
                current_string = current_string[:-1] + chr(ord(current_string[-1]) + 1)
                result.append(current_string)
        
        return result