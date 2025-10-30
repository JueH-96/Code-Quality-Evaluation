class Solution:
    def maximumLength(self, s: str) -> int:
        n = len(s)
        # We'll check all possible lengths L (from maximum possible down to 1)
        # for each letter c, and count how many times c repeated L appears in s.
        # If any such special substring occurs at least thrice, we return L.
        for L in range(n, 0, -1):
            for c in "abcdefghijklmnopqrstuvwxyz":
                pattern = c * L
                count = 0
                # Check all substrings of length L (overlapping allowed)
                for i in range(n - L + 1):
                    if s[i:i+L] == pattern:
                        count += 1
                        if count >= 3:
                            return L
        return -1

# For local testing (these lines can be removed when running on an online judge):
if __name__ == "__main__":
    sol = Solution()
    print(sol.maximumLength("aaaa"))    # Expected Output: 2
    print(sol.maximumLength("abcdef"))  # Expected Output: -1
    print(sol.maximumLength("abcaba"))  # Expected Output: 1