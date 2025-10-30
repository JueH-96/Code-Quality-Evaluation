def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    cards = [(int(next(it)), int(next(it))) for _ in range(n)]
    
    # We'll use a bitmask to represent the set of remaining cards.
    # dp(mask) returns True if the current player (with state 'mask' of remaining cards)
    # can force a win; otherwise, it returns False.
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dp(mask):
        # Build list of indices from the bitmask
        indices = [i for i in range(n) if mask & (1 << i)]
        found_move = False
        # Try all pairs of cards in the state.
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx1 = indices[i]
                idx2 = indices[j]
                # Check if the pair is valid to remove.
                if cards[idx1][0] == cards[idx2][0] or cards[idx1][1] == cards[idx2][1]:
                    found_move = True
                    new_mask = mask & ~(1 << idx1) & ~(1 << idx2)
                    # If the move leads to a losing state for the opponent, this is a winning move.
                    if not dp(new_mask):
                        return True
        # If no valid move exists, the current player loses.
        if not found_move:
            return False
        # If all moves lead to winning positions for the opponent, then current player loses.
        return False

    # full_mask has all cards available.
    full_mask = (1 << n) - 1
    result = "Takahashi" if dp(full_mask) else "Aoki"
    sys.stdout.write(result)

if __name__ == '__main__':
    main()