class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        def color(coordinate):
            column = ord(coordinate[0]) - ord('a') + 1
            row = int(coordinate[1])
            return (column + row) % 2
        return color(coordinate1) == color(coordinate2)