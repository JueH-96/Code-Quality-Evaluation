class Solution:
    def stringSequence(self, target: str) -> List[str]:
        current = ""
        sequence = []
        for i in range(len(target)):
            # Extend the current string by appending 'a' until its length is i + 1
            while len(current) < i + 1:
                current += 'a'
                sequence.append(current)
            # If the last character is not the desired one, use Key 2 to adjust it
            if current[-1] != target[i]:
                # Calculate the difference in positions in the alphabet
                diff = (ord(target[i]) - ord(current[-1])) % 26
                for _ in range(diff):
                    # Change the last character to the next one in the alphabet
                    current = current[:-1] + chr((ord(current[-1]) - ord('a') + 1) % 26 + ord('a'))
                    sequence.append(current)
        return sequence