class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # Convert column letters to 0-7 and row numbers to 0-7 for both coordinates
        col1 = ord(coordinate1[0]) - ord('a')
        row1 = int(coordinate1[1]) - 1
        sum1 = col1 + row1
        
        col2 = ord(coordinate2[0]) - ord('a')
        row2 = int(coordinate2[1]) - 1
        sum2 = col2 + row2
        
        # Check if the sums have the same parity (even or odd)
        return (sum1 % 2) == (sum2 % 2)