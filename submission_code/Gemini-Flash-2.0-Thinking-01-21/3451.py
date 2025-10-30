class Solution:
    def compressedString(self, word: str) -> str:
        comp = ""
        while word:
            char = word[0]
            count = 0
            for c in word:
                if c == char:
                    count += 1
                else:
                    break
            prefix_len = min(count, 9)
            comp += str(prefix_len) + char
            word = word[prefix_len:]
        return comp