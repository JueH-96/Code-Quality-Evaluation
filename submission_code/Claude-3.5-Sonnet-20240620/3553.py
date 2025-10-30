class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        def is_black(coord):
            col = ord(coord[0]) - ord('a')
            row = int(coord[1]) - 1
            return (col + row) % 2 == 0
        
        return is_black(coordinate1) == is_black(coordinate2)