class Solution:
    def kthCharacter(self, k: int) -> str:
        word = "a"
        while len(word) < k:
            next_str = ""
            for c in word:
                if c == 'z':
                    next_str += 'a'
                else:
                    next_str += chr(ord(c) + 1)
            word += next_str
        
        return word[k-1]