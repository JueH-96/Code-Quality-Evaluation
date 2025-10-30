class Solution:
    def compressedString(self, word: str) -> str:
        comp = ""
        n = len(word)
        i = 0
        while i < n:
            j = i
            while j < n and word[j] == word[i]:
                j += 1
            count = j - i
            while count > 0:
                comp += str(min(count, 9))
                comp += word[i]
                count -= 9
            i = j
        return comp