class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # Convert a coordinate like "a1" into a numeric parity:
        # file 'a'->1, 'b'->2, ..., 'h'->8; rank '1'->1, ..., '8'->8.
        def parity(coord: str) -> int:
            file_idx = ord(coord[0]) - ord('a') + 1
            rank_idx = int(coord[1])
            return (file_idx + rank_idx) % 2

        return parity(coordinate1) == parity(coordinate2)