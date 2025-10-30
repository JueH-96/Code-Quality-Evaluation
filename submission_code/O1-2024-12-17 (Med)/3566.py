class Solution:
    def stringSequence(self, target: str) -> List[str]:
        from typing import List
        
        result = []
        # We'll build the target one character at a time.
        # For each character in the target, we first press key1 to append an 'a',
        # then press key2 the required number of times to change 'a' into that character.

        current_string = ""
        
        for ch in target:
            # Press key1 (append 'a')
            current_string += 'a'
            result.append(current_string)
            
            # Press key2 enough times to go from 'a' to the desired character 'ch'
            steps_needed = (ord(ch) - ord('a')) % 26
            for _ in range(steps_needed):
                # Replace the last character with the "next" character
                last_char = current_string[-1]
                next_char = chr((ord(last_char) - ord('a') + 1) % 26 + ord('a'))
                current_string = current_string[:-1] + next_char
                result.append(current_string)
                
        return result