class Solution:
    def compressedString(self, word: str) -> str:
        comp = []
        count = 1
        for i in range(1, len(word) + 1):
            if i < len(word) and word[i] == word[i-1]:
                count += 1
            else:
                comp.append(str(min(count, 9)) + word[i-1])
                count = 1
        return ''.join(comp)