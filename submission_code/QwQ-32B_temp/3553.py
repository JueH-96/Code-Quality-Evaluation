class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        def get_parity(coord):
            col = ord(coord[0]) - ord('a') + 1
            row = int(coord[1])
            return (col + row) % 2
        return get_parity(coordinate1) == get_parity(coordinate2)