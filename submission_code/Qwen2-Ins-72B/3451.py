class Solution:
    def compressedString(self, word: str) -> str:
        if not word:
            return ""
        
        compressed = []
        count = 1
        for i in range(1, len(word)):
            if word[i] == word[i-1]:
                count += 1
            else:
                compressed.append(str(count) + word[i-1])
                count = 1
        compressed.append(str(count) + word[-1])
        
        return "".join(compressed)