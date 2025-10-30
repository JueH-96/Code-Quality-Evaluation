class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        def get_color_sum(coord):
            col = ord(coord[0]) - ord('a') + 1
            row = int(coord[1])
            return col + row
        
        sum1 = get_color_sum(coordinate1)
        sum2 = get_color_sum(coordinate2)
        
        return (sum1 % 2) == (sum2 % 2)