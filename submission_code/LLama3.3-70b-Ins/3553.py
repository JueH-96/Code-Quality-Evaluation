class Solution:
    def squareIsWhite(self, coordinate: str) -> bool:
        # Convert the column letter to a number (a=0, b=1, ..., h=7)
        col = ord(coordinate[0]) - ord('a')
        # Convert the row number to an integer
        row = int(coordinate[1]) - 1
        
        # A square is white if the sum of its row and column indices is even
        return (col + row) % 2 == 0

    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # Check if both squares are white or both are black
        return self.squareIsWhite(coordinate1) == self.squareIsWhite(coordinate2)