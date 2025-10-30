class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        def is_black(coordinate):
            column, row = coordinate[0], coordinate[1]
            return (ord(column) - ord('a') + int(row)) % 2 == 0
        
        return is_black(coordinate1) == is_black(coordinate2)