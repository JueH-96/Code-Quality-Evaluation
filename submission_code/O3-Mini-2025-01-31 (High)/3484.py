class Solution:
    def getSmallestString(self, s: str) -> str:
        # Start with the original string as the best candidate.
        best = s
        n = len(s)
        
        # Try swapping every pair of adjacent digits if they have the same parity.
        for i in range(n - 1):
            # Check if the adjacent digits are both odd or both even.
            if (int(s[i]) % 2) == (int(s[i+1]) % 2):
                # Create a candidate string by swapping these two digits.
                candidate = s[:i] + s[i+1] + s[i] + s[i+2:]
                # Update best if the candidate is lexicographically smaller.
                if candidate < best:
                    best = candidate
        return best

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.getSmallestString("45320"))  # Output: "43520"
    print(sol.getSmallestString("001"))    # Output: "001"