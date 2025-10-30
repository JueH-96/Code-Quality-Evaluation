class Solution:
    def minimumSteps(self, s: str) -> int:
        # We'll compute the number of zeroes to the right of each black ball.
        swaps = 0
        white_count = 0
        # Loop from end to start so that we know how many white balls (0's) come after.
        for char in reversed(s):
            if char == '0':
                white_count += 1
            else:  # char == '1'
                swaps += white_count
        return swaps

# Testing the implementation with the provided examples
if __name__ == "__main__":
    sol = Solution()
    # Example 1
    s = "101"
    print(sol.minimumSteps(s))  # Expected output: 1
    # Example 2
    s = "100"
    print(sol.minimumSteps(s))  # Expected output: 2
    # Example 3
    s = "0111"
    print(sol.minimumSteps(s))  # Expected output: 0