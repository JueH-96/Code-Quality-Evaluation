from typing import List

class Solution:
    def stringSequence(self, target: str) -> List[str]:
        result = []
        current = ""
        
        for char in target:
            # Press key 1: Append 'a'
            current += "a"
            result.append(current)
            
            # Press key 2 until the last character is the target character
            while current[-1] != char:
                last_char = current[-1]
                next_char = chr(((ord(last_char) - ord('a') + 1) % 26) + ord('a'))
                current = current[:-1] + next_char
                result.append(current)
        
        return result