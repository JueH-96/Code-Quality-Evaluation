class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # Extract column and row for the first coordinate
        c1 = ord(coordinate1[0]) - 96
        r1 = int(coordinate1[1])
        sum1 = c1 + r1
        
        # Extract column and row for the second coordinate
        c2 = ord(coordinate2[0]) - 96
        r2 = int(coordinate2[1])
        sum2 = c2 + r2
        
        # Check if both sums have the same parity
        return (sum1 % 2) == (sum2 % 2)