class Solution:
    def countKeyChanges(self, s: str) -> int:
        # Initialize the counter for key changes.
        changes = 0
        # Iterate through the string starting from index 1.
        for i in range(1, len(s)):
            # Check if the lower-case version of the current character 
            # is different from that of the previous character.
            if s[i].lower() != s[i-1].lower():
                changes += 1
        return changes

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.countKeyChanges("aAbBcC"))  # Expected output: 2
    print(sol.countKeyChanges("AaAaAaaA"))  # Expected output: 0