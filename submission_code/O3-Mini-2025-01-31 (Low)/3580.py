class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        n, m = len(s), len(pattern)
        # Iterate over all possible starting indices in s where a substring of length m exists.
        for i in range(n - m + 1):
            diff = 0
            # Check the substring s[i:i+m] against pattern.
            for j in range(m):
                if s[i + j] != pattern[j]:
                    diff += 1
                    # If more than one difference is found, no need to check further.
                    if diff > 1:
                        break
            # If at most one difference, we found the valid index.
            if diff <= 1:
                return i
        # If no such substring exists return -1.
        return -1
                        
# Example usage:
# Uncomment the following lines to test the code locally.
# if __name__ == "__main__":
#     sol = Solution()
#     print(sol.minStartingIndex("abcdefg", "bcdffg"))   # Output: 1
#     print(sol.minStartingIndex("ababbababa", "bacaba"))  # Output: 4
#     print(sol.minStartingIndex("abcd", "dba"))           # Output: -1
#     print(sol.minStartingIndex("dde", "d"))              # Output: 0