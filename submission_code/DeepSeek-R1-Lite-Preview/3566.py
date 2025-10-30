class Solution:
    def stringSequence(self, target: str) -> List[str]:
        sequence = []
        current = ""
        
        def next_char(c):
            if c == 'z':
                return 'a'
            else:
                return chr(ord(c) + 1)
        
        for char in target:
            # Append 'a' and add to sequence
            current += 'a'
            sequence.append(current)
            
            # Calculate the number of Key 2 presses needed
            diff = (ord(char) - ord('a')) % 26
            for _ in range(diff):
                # Modify the last character
                current = current[:-1] + next_char(current[-1])
                sequence.append(current)
        
        return sequence