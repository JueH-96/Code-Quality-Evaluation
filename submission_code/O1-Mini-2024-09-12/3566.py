class Solution:
    def stringSequence(self, target: str) -> List[str]:
        result = []
        current = ""
        for c in target:
            # Press Key 1: append 'a'
            current += 'a'
            result.append(current)
            
            # Calculate the number of Key 2 presses needed
            n = (ord(c) - ord('a')) % 26
            for _ in range(n):
                # Change the last character to the next character in the alphabet
                last_char = current[-1]
                next_char = chr(((ord(last_char) - ord('a') + 1) % 26) + ord('a'))
                current = current[:-1] + next_char
                result.append(current)
                
        return result