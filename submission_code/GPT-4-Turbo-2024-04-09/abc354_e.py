def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    cards = [(int(data[i*2+1]), int(data[i*2+2])) for i in range(N)]
    
    from functools import lru_cache
    
    # State is represented by a bitmask of length N, where the i-th bit is 1 if the i-th card is still in play
    @lru_cache(None)
    def can_win(state):
        # If no moves are possible, the current player loses
        moves_possible = False
        # Try to find any two cards that can be removed
        for i in range(N):
            if not (state & (1 << i)):
                continue
            for j in range(i + 1, N):
                if not (state & (1 << j)):
                    continue
                # Check if cards i and j can be removed
                if (cards[i][0] == cards[j][0] or cards[i][1] == cards[j][1]):
                    # Make the move and remove cards i and j
                    new_state = state & ~(1 << i) & ~(1 << j)
                    # If the opponent cannot win with this new state, then the current player can win
                    if not can_win(new_state):
                        return True
                    moves_possible = True
        
        # If no moves were possible at all, the current player loses
        if not moves_possible:
            return False
        
        # If all possible moves lead to a winning state for the opponent, the current player loses
        return False
    
    # Initial state: all cards are in play
    initial_state = (1 << N) - 1
    
    # Takahashi starts, so we evaluate the initial state for him
    if can_win(initial_state):
        print("Takahashi")
    else:
        print("Aoki")