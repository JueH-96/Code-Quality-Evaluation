class Solution:
    def kthCharacter(self, k: int) -> str:
        word = "a"
        while len(word) < k:
            transformed = []
            for c in word:
                if c == 'z':
                    transformed.append('a')
                else:
                    transformed.append(chr(ord(c) + 1))
            transformed_str = ''.join(transformed)
            word += transformed_str
        return word[k-1]