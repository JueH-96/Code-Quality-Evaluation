class Solution:
    def kthCharacter(self, k: int) -> str:
        word = "a"
        while len(word) < k:
            new_part = []
            for c in word:
                next_c = chr((ord(c) - ord('a') + 1) % 26 + ord('a'))
                new_part.append(next_c)
            word += ''.join(new_part)
        return word[k-1]