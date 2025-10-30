class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # Helper function to determine the color of a square by its coordinate
        def color(coord):
            return ((ord(coord[0]) - ord('a')) + (int(coord[1]) - 1)) % 2
        
        return color(coordinate1) == color(coordinate2)