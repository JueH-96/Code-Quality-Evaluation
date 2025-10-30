class Solution:
    def kthCharacter(self, k: int) -> str:
        word = "a"
        
        # Keep building the string until its length >= k
        while len(word) < k:
            # Create the "next characters" string,
            # where 'z' wraps around to 'a'
            next_part = "".join(
                chr((ord(c) - ord('a') + 1) % 26 + ord('a')) for c in word
            )
            # Append it to the current word
            word += next_part
        
        # Return the k-th character (1-indexed in problem statement)
        return word[k - 1]