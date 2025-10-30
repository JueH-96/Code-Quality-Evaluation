class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # Calculate sum for coordinate1
        col1 = coordinate1[0]
        row1 = coordinate1[1]
        sum1 = (ord(col1) - ord('a')) + int(row1)
        
        # Calculate sum for coordinate2
        col2 = coordinate2[0]
        row2 = coordinate2[1]
        sum2 = (ord(col2) - ord('a')) + int(row2)
        
        # Check if both sums have the same parity
        return (sum1 % 2) == (sum2 % 2)