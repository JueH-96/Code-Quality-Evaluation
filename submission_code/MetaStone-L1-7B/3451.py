class Solution:
    def compressedString(self, word: str) -> str:
        result = []
        current_index = 0
        n = len(word)
        while current_index < n:
            current_char = word[current_index]
            count = 1
            i = current_index + 1
            while i < n and word[i] == current_char:
                count += 1
                i += 1
            take = min(count, 9)
            result.append(str(take))
            result.append(current_char)
            current_index += take
        return ''.join(result)