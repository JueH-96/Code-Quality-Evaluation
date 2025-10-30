class Solution:
    def reverseDegree(self, s: str) -> int:
        total = 0
        # For each character, calculate the reversed rank and multiply by its 1-indexed position in the string.
        for i, ch in enumerate(s, start=1):
            # 'a' is 26 in reversed order, 'b' is 25, ..., 'z' is 1.
            reversed_value = 26 - (ord(ch) - ord('a'))
            total += reversed_value * i
        return total

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Test case 1:
    print(sol.reverseDegree("abc"))  # Expected output: 148
    # Test case 2:
    print(sol.reverseDegree("zaza"))  # Expected output: 160