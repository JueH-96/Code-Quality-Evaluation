class Solution:
    def kthCharacter(self, k: int) -> str:
        word = "a"
        
        # Function to get the "next" string from current word
        # where 'z' -> 'a' and others -> next character
        def next_string(s: str) -> str:
            result = []
            for ch in s:
                # Compute next character in a cyclic manner
                next_ch = chr((ord(ch) - ord('a') + 1) % 26 + ord('a'))
                result.append(next_ch)
            return "".join(result)
        
        # Keep generating until the length is at least k
        while len(word) < k:
            word += next_string(word)
        
        # Return the k-th character (1-indexed)
        return word[k - 1]