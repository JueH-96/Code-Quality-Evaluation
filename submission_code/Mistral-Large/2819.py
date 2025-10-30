class Solution:
    def removeTrailingZeros(self, num: str) -> str:
        # Strip trailing zeros using rstrip method
        return num.rstrip('0')

# Example usage:
# solution = Solution()
# print(solution.removeTrailingZeros("51230100"))  # Output: "512301"
# print(solution.removeTrailingZeros("123"))       # Output: "123"