class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # Define the color of each square on the chessboard
        board = {
            'a': 1, 'b': 0, 'c': 1, 'd': 0, 'e': 1, 'f': 0, 'g': 1, 'h': 0,
            '1': 0, '2': 1, '3': 0, '4': 1, '5': 0, '6': 1, '7': 0, '8': 1
        }
        
        # Check if the colors of the two squares are the same
        return board[coordinate1[0]] == board[coordinate1[1]] == board[coordinate2[0]] == board[coordinate2[1]]