class Solution:
    def getSmallestString(self, s: str) -> str:
        best = s
        n = len(s)
        for i in range(n - 1):
            # check if s[i] and s[i+1] have the same parity
            if (int(s[i]) % 2) == (int(s[i+1]) % 2):
                # swap them
                candidate = s[:i] + s[i+1] + s[i] + s[i+2:]
                # keep the lexicographically smallest
                if candidate < best:
                    best = candidate
        return best


# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.getSmallestString("45320"))  # "43520"
    print(sol.getSmallestString("001"))    # "001"