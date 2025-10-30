class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # Calculate the color of the squares based on their coordinates
        # The column 'a' to 'h' corresponds to 1 to 8
        # The row '1' to '8' corresponds to 1 to 8
        col1 = ord(coordinate1[0]) - ord('a') + 1
        row1 = int(coordinate1[1])
        col2 = ord(coordinate2[0]) - ord('a') + 1
        row2 = int(coordinate2[1])
        
        # Determine the color of the squares
        color1 = (col1 + row1) % 2
        color2 = (col2 + row2) % 2
        
        return color1 == color2