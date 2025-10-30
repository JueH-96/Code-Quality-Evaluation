class Solution:
    def kthCharacter(self, k: int) -> str:
        word = "a"
        
        while len(word) < k:
            # Build the next string by converting each character to its next alphabet
            next_str = []
            for ch in word:
                if ch == 'z':
                    next_str.append('a')
                else:
                    next_str.append(chr(ord(ch) + 1))
            
            word += "".join(next_str)
        
        return word[k - 1]