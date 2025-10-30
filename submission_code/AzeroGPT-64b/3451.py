class Solution:
    def compressedString(self, word: str) -> str:
        comp = []
        count = 1
        for i in range(len(word)):
            if i < len(word) - 1 and word[i] == word[i + 1]:
                count += 1
            else:
                comp.append(str(count) + word[i])
                count = 1
        return ''.join(comp)