def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    A = list(map(int, data))
    
    # Since the grid is 3x3, we can hardcode the indices for rows, columns, and diagonals
    rows = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    cols = [(0, 3, 6), (1, 4, 7), (2, 5, 8)]
    diags = [(0, 4, 8), (2, 4, 6)]
    
    # Check all possible winning lines
    lines = rows + cols + diags
    
    # We will use minimax with memoization to determine the optimal play
    from functools import lru_cache
    
    @lru_cache(None)
    def minimax(state, turn):
        # State is a tuple of 9 elements, each can be 0 (empty), 1 (Takahashi), or 2 (Aoki)
        # Turn is 1 for Takahashi, -1 for Aoki
        
        # Check for a winner
        for line in lines:
            if state[line[0]] == state[line[1]] == state[line[2]] != 0:
                if state[line[0]] == 1:
                    return float('inf')  # Takahashi wins
                else:
                    return float('-inf')  # Aoki wins
        
        # Check if the board is full
        if all(s != 0 for s in state):
            # Calculate scores based on the board state
            takahashi_score = sum(A[i] for i in range(9) if state[i] == 1)
            aoki_score = sum(A[i] for i in range(9) if state[i] == 2)
            if takahashi_score > aoki_score:
                return float('inf')
            else:
                return float('-inf')
        
        # Minimax logic
        best_score = float('-inf') if turn == 1 else float('inf')
        
        for i in range(9):
            if state[i] == 0:
                new_state = list(state)
                new_state[i] = 1 if turn == 1 else 2
                score = minimax(tuple(new_state), -turn)
                if turn == 1:
                    best_score = max(best_score, score + A[i])
                else:
                    best_score = min(best_score, score - A[i])
        
        return best_score
    
    # Start the game with an empty board and Takahashi's turn
    initial_state = (0,) * 9
    result = minimax(initial_state, 1)
    
    if result > 0:
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == "__main__":
    main()