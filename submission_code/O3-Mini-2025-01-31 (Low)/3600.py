class Solution:
    def kthCharacter(self, k: int) -> str:
        # start with initial string
        word = "a"
        # while the length of word is less than k, perform the operation
        while len(word) < k:
            # Create newString by replacing every character with its next in alphabet (wrap z->a)
            new_str = []
            for ch in word:
                # Calculate next char: 'a' = 97, ..., 'z' = 122
                # Next char = (current_char_index + 1) mod 26 
                next_char = chr((ord(ch) - ord('a') + 1) % 26 + ord('a'))
                new_str.append(next_char)
            # Convert list to string
            new_str = ''.join(new_str)
            # Append new string to word
            word += new_str
        # Return the k^th character (1-indexed)
        return word[k - 1]
        
# Example usage:
# Uncomment below to run a simple test.
if __name__ == "__main__":
    sol = Solution()
    print(sol.kthCharacter(5))   # Expected output: "b"
    print(sol.kthCharacter(10))  # Expected output: "c"