class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # Calculate sum for coordinate1
        col1 = ord(coordinate1[0]) - ord('a') + 1
        row1 = int(coordinate1[1])
        sum1 = col1 + row1
        
        # Calculate sum for coordinate2
        col2 = ord(coordinate2[0]) - ord('a') + 1
        row2 = int(coordinate2[1])
        sum2 = col2 + row2
        
        # Check if both sums have the same parity
        return (sum1 % 2) == (sum2 % 2)