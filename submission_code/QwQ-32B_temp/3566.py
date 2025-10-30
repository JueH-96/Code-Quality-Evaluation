class Solution:
    def stringSequence(self, target: str) -> List[str]:
        current = []  # Using a list for efficient modifications
        output = []
        for char in target:
            # Press Key 1: append 'a'
            current.append('a')
            output.append(''.join(current))
            
            # Calculate the number of Key 2 presses needed
            target_ord = ord(char)
            steps = (target_ord - ord('a')) % 26
            
            for _ in range(steps):
                # Press Key 2: increment the last character
                last_char = current[-1]
                last_ord = ord(last_char)
                new_ord = (last_ord - ord('a') + 1) % 26 + ord('a')
                current[-1] = chr(new_ord)
                output.append(''.join(current))
        
        return output