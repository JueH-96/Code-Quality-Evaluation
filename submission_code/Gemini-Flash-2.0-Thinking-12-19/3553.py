class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        def get_square_color(coordinate):
            col_char = coordinate[0]
            row_char = coordinate[1]
            col_index = ord(col_char) - ord('a') + 1
            row_index = int(row_char)
            if (col_index + row_index) % 2 == 0:
                return "black"
            else:
                return "white"

        color1 = get_square_color(coordinate1)
        color2 = get_square_color(coordinate2)
        return color1 == color2