class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        def get_color(coordinate):
            col = ord(coordinate[0]) - ord('a') + 1
            row = int(coordinate[1])
            return (col + row) % 2
        
        return get_color(coordinate1) == get_color(coordinate2)