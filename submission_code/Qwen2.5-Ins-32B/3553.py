class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        def is_same_color(coord1, coord2):
            # Convert the letter to a number (a=1, b=2, ..., h=8)
            col1, col2 = ord(coord1[0]) - ord('a') + 1, ord(coord2[0]) - ord('a') + 1
            row1, row2 = int(coord1[1]), int(coord2[1])
            
            # Check if both squares are on the same color
            return (col1 + row1) % 2 == (col2 + row2) % 2
        
        return is_same_color(coordinate1, coordinate2)