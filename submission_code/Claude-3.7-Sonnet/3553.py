class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # Convert chess notation to numerical indices
        col1 = ord(coordinate1[0]) - ord('a') + 1  # 'a' -> 1, 'b' -> 2, etc.
        row1 = int(coordinate1[1])
        
        col2 = ord(coordinate2[0]) - ord('a') + 1
        row2 = int(coordinate2[1])
        
        # Check if both squares have the same color by comparing parity
        return (row1 + col1) % 2 == (row2 + col2) % 2