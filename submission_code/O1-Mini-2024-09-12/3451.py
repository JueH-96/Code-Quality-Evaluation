class Solution:
    def compressedString(self, word: str) -> str:
        if not word:
            return ""
        comp = []
        current_char = word[0]
        count = 1
        for c in word[1:]:
            if c == current_char and count < 9:
                count += 1
            else:
                comp.append(str(count))
                comp.append(current_char)
                current_char = c
                count = 1
        # Append the last run
        comp.append(str(count))
        comp.append(current_char)
        return ''.join(comp)