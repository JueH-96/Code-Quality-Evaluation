class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        col1 = ord(coordinate1[0]) - ord('a')
        row1 = int(coordinate1[1]) -1
        col2 = ord(coordinate2[0]) - ord('a')
        row2 = int(coordinate2[1]) -1

        if (col1 + row1) % 2 == (col2 + row2) % 2:
            return True
        else:
            return False