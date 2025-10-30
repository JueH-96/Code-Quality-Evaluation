class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # Convert the column letters to numerical values and get row numbers
        col1 = ord(coordinate1[0]) - ord('a') + 1
        row1 = int(coordinate1[1])
        col2 = ord(coordinate2[0]) - ord('a') + 1
        row2 = int(coordinate2[1])
        
        # Calculate the sum of column and row for each coordinate
        sum1 = col1 + row1
        sum2 = col2 + row2
        
        # Check if both sums are even or both are odd
        return (sum1 % 2) == (sum2 % 2)