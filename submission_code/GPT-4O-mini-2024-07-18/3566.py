from typing import List

class Solution:
    def stringSequence(self, target: str) -> List[str]:
        result = []
        current_string = ""
        
        for char in target:
            # Append 'a' to the current string
            while len(current_string) < ord(char) - ord('a') + 1:
                current_string += 'a'
                result.append(current_string)
            
            # Change the last character to the target character
            while current_string[-1] < char:
                current_string = current_string[:-1] + chr(ord(current_string[-1]) + 1)
                result.append(current_string)
        
        return result