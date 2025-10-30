from typing import List

class Solution:
    def stringSequence(self, target: str) -> List[str]:
        result = []
        current = ""
        
        # Process the first character
        # Must press key 1: append 'a'
        current += "a"
        result.append(current)
        # Now, if needed, press key 2 until last character equals target[0]
        while current[-1] != target[0]:
            # Update last character 
            # Compute next character in cycle: next of current[-1]
            next_char = chr(((ord(current[-1]) - ord('a') + 1) % 26) + ord('a'))
            current = current[:-1] + next_char
            result.append(current)
            
        # Process remaining characters in target
        for ch in target[1:]:
            # Press key 1: append 'a'
            current += "a"
            result.append(current)
            # Now, increment the last character until it becomes ch
            while current[-1] != ch:
                next_char = chr(((ord(current[-1]) - ord('a') + 1) % 26) + ord('a'))
                current = current[:-1] + next_char
                result.append(current)
                
        return result