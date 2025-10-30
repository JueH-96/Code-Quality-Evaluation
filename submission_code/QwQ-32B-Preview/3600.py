class Solution:
    def kthCharacter(self, k: int) -> str:
        def shift(s):
            shifted = []
            for c in s:
                if c == 'z':
                    shifted.append('a')
                else:
                    shifted.append(chr(ord(c) + 1))
            return ''.join(shifted)
        
        word = "a"
        while len(word) < k:
            new_string = shift(word)
            word += new_string
        return word[k - 1]