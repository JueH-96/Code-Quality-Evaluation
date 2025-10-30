class Solution:
    def kthCharacter(self, k: int) -> str:
        word = 'a'
        while len(word) < k:
            new_part = []
            for c in word:
                shifted = chr((ord(c) - ord('a') + 1) % 26 + ord('a'))
                new_part.append(shifted)
            word += ''.join(new_part)
        return word[k-1]