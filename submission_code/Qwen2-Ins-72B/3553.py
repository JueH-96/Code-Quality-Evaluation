class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # Calculate the sum of the ASCII values of the coordinates
        sum1 = ord(coordinate1[0]) + int(coordinate1[1])
        sum2 = ord(coordinate2[0]) + int(coordinate2[1])
        
        # If the sums are both even or both odd, the squares have the same color
        return (sum1 % 2 == sum2 % 2)