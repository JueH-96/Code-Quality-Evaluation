class Solution:
    def kthCharacter(self, k: int) -> str:
        # Start with the initial word
        word = "a"
        
        # Keep expanding until we have at least k characters
        while len(word) < k:
            # Build the transformed string
            transformed = []
            for c in word:
                # Compute next character in a cyclic manner
                next_char = chr((ord(c) - ord('a') + 1) % 26 + ord('a'))
                transformed.append(next_char)
            # Append to the original word
            word += "".join(transformed)
        
        # Return the k-th character (1-based index)
        return word[k-1]

# Example usage:
# sol = Solution()
# print(sol.kthCharacter(5))  # Output: "b"
# print(sol.kthCharacter(10)) # Output: "c"