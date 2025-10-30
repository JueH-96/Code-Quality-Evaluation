class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # Calculate color: convert letter ('a'-'h') to 1-8 and number
        def color(coordinate):
            col = ord(coordinate[0]) - ord('a') + 1
            row = int(coordinate[1])
            # if sum is even, one color, if odd, the opposite color
            return (col + row) % 2
        
        return color(coordinate1) == color(coordinate2)


# Example test cases:

sol = Solution()
print(sol.checkTwoChessboards("a1", "c3"))  # Expected output: True (both are black)
print(sol.checkTwoChessboards("a1", "h3"))  # Expected output: False