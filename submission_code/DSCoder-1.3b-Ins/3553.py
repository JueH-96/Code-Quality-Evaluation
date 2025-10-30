class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # Check if the squares are the same color
        if (ord(coordinate1[0]) + int(coordinate1[1])) % 2 == (ord(coordinate2[0]) + int(coordinate2[1])) % 2:
            return True
        else:
            return False