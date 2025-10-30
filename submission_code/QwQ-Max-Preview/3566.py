class Solution:
    def stringSequence(self, target: str) -> List[str]:
        current = ""
        output = []
        n = len(target)
        for i in range(n):
            required_length = i + 1
            # Append 'a's until the current length reaches the required length
            while len(current) < required_length:
                current += 'a'
                output.append(current)
            # Calculate delta for the current character to target character
            current_char = current[-1]
            required_char = target[i]
            delta = (ord(required_char) - ord(current_char)) % 26
            # Modify the last character delta times
            for _ in range(delta):
                new_last_char = chr((ord(current[-1]) - ord('a') + 1) % 26 + ord('a'))
                current = current[:-1] + new_last_char
                output.append(current)
        return output