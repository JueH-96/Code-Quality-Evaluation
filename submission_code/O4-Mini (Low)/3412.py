class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        # Build a mapping from character to its index in t
        pos_in_t = {ch: idx for idx, ch in enumerate(t)}
        
        # Sum the absolute differences of indices for each character in s
        diff = 0
        for idx_s, ch in enumerate(s):
            idx_t = pos_in_t[ch]
            diff += abs(idx_s - idx_t)
        
        return diff


# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1
    print(sol.findPermutationDifference("abc", "bac"))   # Output: 2
    # Example 2
    print(sol.findPermutationDifference("abcde", "edbac"))  # Output: 12