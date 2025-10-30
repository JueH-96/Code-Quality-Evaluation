def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    AB = [(int(input_data[2*i+1]), int(input_data[2*i+2])) for i in range(N)]
    
    # Precompute which pairs of cards can be removed together
    # i.e., they share the same front (A) or the same back (B).
    removable_pairs = []
    for i in range(N):
        for j in range(i+1, N):
            if AB[i][0] == AB[j][0] or AB[i][1] == AB[j][1]:
                removable_pairs.append((i, j))
    
    # We'll use a bitmask to represent which cards remain on the table.
    # A state s is winning if the current player can force a win from that state.
    from functools import lru_cache
    
    @lru_cache(None)
    def can_win(state):
        # If no move is possible, this is a losing position for the current player.
        # We'll check for any valid pair removals.
        # If there's a move leading the opponent to a losing position, then this is winning.
        
        # Try all removable pairs that are still available in 'state'.
        # If none is available, the current player loses.
        for (i, j) in removable_pairs:
            if (state & (1 << i)) and (state & (1 << j)):
                # Remove cards i and j from state
                new_state = state & ~(1 << i) & ~(1 << j)
                if not can_win(new_state):
                    # If the opponent loses from new_state, we win.
                    return True
        
        # If all moves lead to positions where the opponent can win,
        # or if there are no moves, we lose from this state.
        return False
    
    # Start with all cards present.
    full_state = (1 << N) - 1
    
    # If can_win(full_state) is True, Takahashi (the first player) wins.
    # Otherwise, Aoki wins.
    if can_win(full_state):
        print("Takahashi")
    else:
        print("Aoki")

def main():
    solve()

# Call solve() after definition
if __name__ == "__main__":
    main()