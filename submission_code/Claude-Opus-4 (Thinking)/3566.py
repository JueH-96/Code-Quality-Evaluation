class Solution:
    def stringSequence(self, target: str) -> List[str]:
        result = []
        current = ""
        
        for i in range(len(target)):
            # Press Key 1 to append 'a'
            current += 'a'
            result.append(current)
            
            # Press Key 2 to change last character to target[i]
            while current[-1] != target[i]:
                # Change last character to next in alphabet
                last_char = current[-1]
                next_char = chr((ord(last_char) - ord('a') + 1) % 26 + ord('a'))
                current = current[:-1] + next_char
                result.append(current)
        
        return result