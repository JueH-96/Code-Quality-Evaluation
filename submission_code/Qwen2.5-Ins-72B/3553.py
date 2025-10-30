class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        def is_black(coordinate):
            col = ord(coordinate[0]) - ord('a')
            row = int(coordinate[1]) - 1
            return (col + row) % 2 == 0
        
        return is_black(coordinate1) == is_black(coordinate2)