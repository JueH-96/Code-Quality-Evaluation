class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # Calculate sum of column index and row number for coordinate1
        sum1 = (ord(coordinate1[0]) - ord('a')) + int(coordinate1[1])
        # Calculate sum of column index and row number for coordinate2
        sum2 = (ord(coordinate2[0]) - ord('a')) + int(coordinate2[1])
        # Check if both sums have the same parity
        return (sum1 % 2) == (sum2 % 2)