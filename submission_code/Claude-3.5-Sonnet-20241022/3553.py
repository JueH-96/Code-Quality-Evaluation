class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # Convert letter to number (a=0, b=1, etc)
        col1 = ord(coordinate1[0]) - ord('a')
        col2 = ord(coordinate2[0]) - ord('a')
        
        # Convert string number to int
        row1 = int(coordinate1[1]) - 1
        row2 = int(coordinate2[1]) - 1
        
        # A square is black if sum of row and col is even
        # A square is white if sum of row and col is odd
        # So squares have same color if both sums are even or both are odd
        return (col1 + row1) % 2 == (col2 + row2) % 2