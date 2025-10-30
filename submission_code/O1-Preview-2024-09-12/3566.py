class Solution:
    def stringSequence(self, target: str) -> List[str]:
        current_screen = ''
        output = []
        for i in range(len(target)):
            # Append 'a's until we reach required length
            while len(current_screen) < i+1:
                current_screen += 'a'
                output.append(current_screen)
            # Adjust the last character to match target[i]
            current_char = current_screen[-1]
            delta = (ord(target[i]) - ord(current_char)) % 26
            for _ in range(delta):
                # Calculate next character, wrapping from 'z' to 'a'
                current_char_code = (ord(current_screen[-1]) - ord('a') + 1) % 26
                current_char = chr(current_char_code + ord('a'))
                current_screen = current_screen[:-1] + current_char
                output.append(current_screen)
        return output