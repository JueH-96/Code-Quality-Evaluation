class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # Helper to compute parity of a square: even for black, odd for white (or vice versa)
        def parity(coordinate: str) -> int:
            col, row = coordinate[0], coordinate[1]
            # Convert 'a'..'h' to 1..8, add the row number, then take modulo 2
            return ((ord(col) - ord('a') + 1) + int(row)) % 2

        return parity(coordinate1) == parity(coordinate2)