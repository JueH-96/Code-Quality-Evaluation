class Solution:
    def kthCharacter(self, k: int) -> str:
        word = "a"
        while len(word) < k:
            next_str = []
            for c in word:
                if c == 'z':
                    next_str.append('a')
                else:
                    next_str.append(chr(ord(c) + 1))
            word += ''.join(next_str)
        return word[k - 1]