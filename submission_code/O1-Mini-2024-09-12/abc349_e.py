# YOUR CODE HERE
import sys
from functools import lru_cache

def main():
    import sys
    sys.setrecursionlimit(1000000)
    A = []
    for _ in range(3):
        A.extend(map(int, sys.stdin.readline().split()))
    
    lines = [
        (0,1,2),
        (3,4,5),
        (6,7,8),
        (0,3,6),
        (1,4,7),
        (2,5,8),
        (0,4,8),
        (2,4,6)
    ]
    
    @lru_cache(None)
    def dfs(board, is_takahashi_turn, tak_score, aok_score):
        # Check for sequences
        for line in lines:
            colors = [board[i] for i in line]
            if all(c == 'R' for c in colors):
                return 'Takahashi'
            if all(c == 'B' for c in colors):
                return 'Aoki'
        
        # Check if no white cells
        if 'W' not in board:
            if tak_score > aok_score:
                return 'Takahashi'
            else:
                return 'Aoki'
        
        if is_takahashi_turn:
            winner = 'Aoki'
            for i in range(9):
                if board[i] == 'W':
                    new_board = list(board)
                    new_board[i] = 'R'
                    new_tak = tak_score + A[i]
                    outcome = dfs(tuple(new_board), False, new_tak, aok_score)
                    if outcome == 'Takahashi':
                        return 'Takahashi'
            return winner
        else:
            winner = 'Takahashi'
            for i in range(9):
                if board[i] == 'W':
                    new_board = list(board)
                    new_board[i] = 'B'
                    new_aok = aok_score + A[i]
                    outcome = dfs(tuple(new_board), True, tak_score, new_aok)
                    if outcome == 'Aoki':
                        return 'Aoki'
            return winner
    
    initial_board = tuple(['W']*9)
    result = dfs(initial_board, True, 0, 0)
    print(result)

if __name__ == "__main__":
    main()