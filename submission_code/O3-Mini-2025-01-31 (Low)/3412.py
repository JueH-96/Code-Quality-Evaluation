class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        # Create a mapping of character -> index in t for quick lookup
        t_index = {char: idx for idx, char in enumerate(t)}
        total_diff = 0
        
        # Iterate over s and sum the absolute differences with corresponding indices in t
        for idx, char in enumerate(s):
            total_diff += abs(idx - t_index[char])
        
        return total_diff

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1
    s1, t1 = "abc", "bac"
    print(sol.findPermutationDifference(s1, t1))  # Expected output: 2

    # Example 2
    s2, t2 = "abcde", "edbac"
    print(sol.findPermutationDifference(s2, t2))  # Expected output: 12