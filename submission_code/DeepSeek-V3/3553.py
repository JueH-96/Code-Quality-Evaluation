class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        def get_color(coordinate):
            x = ord(coordinate[0]) - ord('a')
            y = int(coordinate[1]) - 1
            return (x + y) % 2 == 0
        
        color1 = get_color(coordinate1)
        color2 = get_color(coordinate2)
        
        return color1 == color2