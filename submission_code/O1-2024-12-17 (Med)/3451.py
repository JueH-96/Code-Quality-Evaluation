class Solution:
    def compressedString(self, word: str) -> str:
        if not word:
            return ""
        
        comp = []
        current_char = word[0]
        count = 1
        
        for i in range(1, len(word)):
            if word[i] == current_char and count < 9:
                count += 1
            else:
                comp.append(str(count) + current_char)
                current_char = word[i]
                count = 1
        
        # Append the last group
        comp.append(str(count) + current_char)
        
        return "".join(comp)