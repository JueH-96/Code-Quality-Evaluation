class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # Helper to get the parity of a square: 0 for black, 1 for white
        def color_parity(coord: str) -> int:
            # file letter 'a'..'h' -> 1..8
            file_num = ord(coord[0]) - ord('a') + 1
            # rank digit '1'..'8' -> 1..8
            rank_num = int(coord[1])
            return (file_num + rank_num) % 2

        return color_parity(coordinate1) == color_parity(coordinate2)