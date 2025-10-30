class Solution:
    def compressedString(self, word: str) -> str:
        comp = []
        if not word:
            return ""
        current_char = word[0]
        count = 1
        for char in word[1:]:
            if char == current_char and count < 9:
                count += 1
            else:
                comp.append(str(count) + current_char)
                current_char = char
                count = 1
        comp.append(str(count) + current_char)
        return ''.join(comp)