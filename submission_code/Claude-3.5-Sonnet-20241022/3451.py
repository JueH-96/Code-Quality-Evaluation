class Solution:
    def compressedString(self, word: str) -> str:
        comp = []
        i = 0
        while i < len(word):
            curr_char = word[i]
            count = 1
            
            # Count consecutive occurrences of current character
            while i + 1 < len(word) and word[i + 1] == curr_char and count < 9:
                count += 1
                i += 1
            
            # Append count and character to result
            comp.append(str(count))
            comp.append(curr_char)
            i += 1
            
        return "".join(comp)