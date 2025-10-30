class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # Convert column letter to number (a=1, b=2, ..., h=8)
        col1 = ord(coordinate1[0]) - 96
        row1 = int(coordinate1[1])
        col2 = ord(coordinate2[0]) - 96
        row2 = int(coordinate2[1])
        
        # Calculate sums
        sum1 = col1 + row1
        sum2 = col2 + row2
        
        # Check if both sums are even or both are odd
        return sum1 % 2 == sum2 % 2