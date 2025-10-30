class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        def get_color(coord):
            col = ord(coord[0]) - ord('a')
            row = int(coord[1]) - 1
            return (col + row) % 2 == 0

        return get_color(coordinate1) == get_color(coordinate2)