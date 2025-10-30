class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        # Count the occurrences of character 'c' in s.
        count = s.count(c)
        # The number of substrings that start and end with 'c' is given by choosing two 
        # occurrences (or the same one twice for a substring of length one), which is:
        return count * (count + 1) // 2

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.countSubstrings("abada", "a"))  # Output: 6
    # Example 2:
    print(sol.countSubstrings("zzz", "z"))  # Output: 6