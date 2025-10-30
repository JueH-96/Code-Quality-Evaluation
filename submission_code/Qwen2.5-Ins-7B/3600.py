class Solution:
    def kthCharacter(self, k: int) -> str:
        word = "a"
        while len(word) < k:
            word += chr((ord(word[-1]) - ord('a') + 1) % 26 + ord('a'))
        return word[k-1]