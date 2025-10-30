class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        def get_color(coordinate):
            column = ord(coordinate[0]) - ord('a')
            row = int(coordinate[1]) - 1
            return (column + row) % 2 == 0

        return get_color(coordinate1) == get_color(coordinate2)