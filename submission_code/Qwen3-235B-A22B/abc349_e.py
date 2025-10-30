import sys
from functools import lru_cache

def main():
    A = []
    for _ in range(3):
        A += list(map(int, sys.stdin.readline().split()))
    
    lines = [ (0,1,2), (3,4,5), (6,7,8),
              (0,3,6), (1,4,7), (2,5,8),
              (0,4,8), (2,4,6) ]
    
    @lru_cache(maxsize=None)
    def dfs(board):
        # Check if any player has a winning line
        for line in lines:
            a, b, c = line
            x, y, z = board[a], board[b], board[c]
            if x == y == z and x != 0:
                winner = 'Takahashi' if x == 1 else 'Aoki'
                return winner
        
        # Check if all cells are filled
        if all(cell != 0 for cell in board):
            t_score = sum(A[i] for i in range(9) if board[i] == 1)
            a_score = sum(A[i] for i in range(9) if board[i] == 2)
            return 'Takahashi' if t_score > a_score else 'Aoki'
        
        # Determine current player
        filled_count = sum(1 for cell in board if cell != 0)
        current = 'Takahashi' if filled_count % 2 == 0 else 'Aoki'
        current_color = 1 if current == 'Takahashi' else 2
        
        other = 'Aoki' if current == 'Takahashi' else 'Takahashi'
        
        # Try all possible moves
        for i in range(9):
            if board[i] == 0:
                # Create new board after move
                new_board = list(board)
                new_board[i] = current_color
                new_board_tuple = tuple(new_board)
                
                # Check if this move creates a winning line
                winner_line = None
                for line in lines:
                    a, b, c = line
                    x, y, z = new_board_tuple[a], new_board_tuple[b], new_board_tuple[c]
                    if x == y == z and x != 0:
                        winner_line = 'Takahashi' if x == 1 else 'Aoki'
                        break
                if winner_line == current:
                    return current
                
                # Check if all filled
                if all(cell != 0 for cell in new_board_tuple):
                    t_score = sum(A[i] for i in range(9) if new_board_tuple[i] == 1)
                    a_score = sum(A[i] for i in range(9) if new_board_tuple[i] == 2)
                    winner = 'Takahashi' if t_score > a_score else 'Aoki'
                    if winner == current:
                        return current
                    continue
                
                # Recursive call
                res = dfs(new_board_tuple)
                if res == current:
                    return current
        
        # No winning move found
        return other
    
    initial_board = tuple([0]*9)
    print(dfs(initial_board))

if __name__ == "__main__":
    main()