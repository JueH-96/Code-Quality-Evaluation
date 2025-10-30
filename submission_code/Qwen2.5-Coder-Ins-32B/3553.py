class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        def is_black(coordinate: str) -> bool:
            col, row = coordinate[0], coordinate[1]
            return (ord(col) + int(row)) % 2 == 1
        
        return is_black(coordinate1) == is_black(coordinate2)