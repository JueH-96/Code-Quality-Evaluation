class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        def get_color(coordinate):
            col = ord(coordinate[0]) - ord('a') + 1  # Convert 'a'-'h' to 1-8
            row = int(coordinate[1])  # Convert '1'-'8' to 1-8
            return (col + row) % 2  # 0 for one color, 1 for another
        
        return get_color(coordinate1) == get_color(coordinate2)