class Solution:
    def stringSequence(self, target: str) -> List[str]:
        output = []
        current = []
        for i in range(len(target)):
            # Ensure the current string has length i+1 by pressing K1 if needed
            while len(current) < i + 1:
                current.append('a')
                output.append(''.join(current))
            # Now, current[i] needs to be changed to target[i] using K2 presses
            target_char = target[i]
            current_char = current[i]
            steps = (ord(target_char) - ord(current_char)) % 26
            # Apply the steps
            for _ in range(steps):
                # Increment the current character
                next_char = chr(ord(current[i]) + 1) if current[i] != 'z' else 'a'
                current[i] = next_char
                output.append(''.join(current))
        return output