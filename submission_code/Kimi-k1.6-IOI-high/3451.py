class Solution:
    def compressedString(self, word: str) -> str:
        result = []
        n = len(word)
        i = 0
        while i < n:
            current_char = word[i]
            count = 0
            while i + count < n and word[i + count] == current_char:
                count += 1
            chunk = min(count, 9)
            result.append(str(chunk))
            result.append(current_char)
            i += chunk
        return ''.join(result)