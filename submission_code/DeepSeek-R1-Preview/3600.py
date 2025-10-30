class Solution:
    def kthCharacter(self, k: int) -> str:
        word = 'a'
        while len(word) < k:
            shifted = ''.join([chr((ord(c) - ord('a') + 1) % 26 + ord('a')) for c in word])
            word += shifted
        return word[k-1]