class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # Process the first coordinate
        col1 = ord(coordinate1[0]) - ord('a') + 1
        row1 = int(coordinate1[1])
        sum1 = col1 + row1
        
        # Process the second coordinate
        col2 = ord(coordinate2[0]) - ord('a') + 1
        row2 = int(coordinate2[1])
        sum2 = col2 + row2
        
        # Check if both sums are even or both are odd
        return (sum1 % 2) == (sum2 % 2)