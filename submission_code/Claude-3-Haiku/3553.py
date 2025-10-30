class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # Convert the coordinates to row and column indices
        row1, col1 = int(coordinate1[1]) - 1, ord(coordinate1[0]) - ord('a')
        row2, col2 = int(coordinate2[1]) - 1, ord(coordinate2[0]) - ord('a')
        
        # Check if the squares have the same color
        if (row1 + col1) % 2 == (row2 + col2) % 2:
            return True
        else:
            return False