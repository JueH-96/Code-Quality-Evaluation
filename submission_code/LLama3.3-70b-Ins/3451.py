class Solution:
    def compressedString(self, word: str) -> str:
        if not word:
            return ""

        comp = ""
        count = 1

        for i in range(1, len(word)):
            if word[i] == word[i - 1]:
                count += 1
            else:
                comp += str(min(count, 9)) + word[i - 1]
                count = 1

        comp += str(min(count, 9)) + word[-1]

        return comp