def solve():
    import sys
    sys.setrecursionlimit(10**7)

    # Read inputs
    A = []
    for _ in range(3):
        A.extend(list(map(int, sys.stdin.readline().split())))
    # A is now a length-9 list in row-major order:
    # index = 0..8 corresponds to:
    # 0->(0,0), 1->(0,1), 2->(0,2), 3->(1,0), 4->(1,1), 5->(1,2), 6->(2,0), 7->(2,1), 8->(2,2)

    # Precompute the sum of all cells:
    total_sum = sum(A)  # This is guaranteed to be odd.

    # Define winning masks for any 3 in a row (rows, cols, diagonals)
    winning_masks = [
        0b111000000,  # row 0
        0b000111000,  # row 1
        0b000000111,  # row 2
        0b100100100,  # col 0
        0b010010010,  # col 1
        0b001001001,  # col 2
        0b100010001,  # diag left-top to right-bottom
        0b001010100,  # diag right-top to left-bottom
    ]

    def has_three_in_a_row(mask):
        # Return True if mask has any 3-in-a-row
        for wm in winning_masks:
            if (mask & wm) == wm:
                return True
        return False

    # We'll do a DFS (backtracking) to see who wins with perfect play.
    # turn == 0 => Takahashi's turn, turn == 1 => Aoki's turn
    # Tmask, Amask are bitmasks of which squares T or A has taken.
    # Tscore, Ascore track the total points each has so far.
    # usedmask bits = 1 if the cell is already taken, 0 if it's free.
    # We'll return 'T' if from this state, with optimal play, Takahashi eventually wins,
    # or 'A' otherwise.

    # A small optimization: We do a depth-limited search with alpha-beta style early returns.
    # Because no matter how large the search is, we can attempt to prune aggressively:
    # - If at T's turn we see a move that forces T to win, we immediately return 'T'.
    # - If at A's turn we see a move that forces A to win, we immediately return 'A'.

    def dfs(Tmask, Amask, Tscore, Ascore, usedmask, turn):
        # Check immediate 3-in-a-row for the previous mover:
        # (Because we set the chosen cell before calling dfs for the next player,
        #  so if there's a new 3-in-a-row, that means the last move created it.)
        if turn == 0:
            # That means it's T's turn NOW, so Aoki was the one who moved just before this call.
            if has_three_in_a_row(Amask):
                return 'A'
        else:
            # It's Aoki's turn, so Takahashi was the one who just moved
            if has_three_in_a_row(Tmask):
                return 'T'

        # Count how many cells used
        used_count = bin(usedmask).count('1')
        if used_count == 9:
            # No cells left, decide by final scores
            if Tscore > Ascore:
                return 'T'
            else:
                return 'A'

        # Otherwise, we can still pick a cell
        # If it's turn == 0, Takahashi picks
        if turn == 0:
            # Try all free squares
            best_outcome = 'A'  # If we cannot find a winning route for T, we'll remain 'A'
            for pos in range(9):
                if not (usedmask & (1 << pos)):
                    # This cell is free
                    # T picks it
                    new_Tmask = Tmask | (1 << pos)
                    new_Tscore = Tscore + A[pos]
                    new_usedmask = usedmask | (1 << pos)
                    # Recurse to next turn
                    outcome = dfs(new_Tmask, Amask, new_Tscore, Ascore, new_usedmask, 1)
                    if outcome == 'T':
                        return 'T'  # Takahashi can force a T outcome, so pick it and prune
                    # Otherwise outcome == 'A', so keep track in best_outcome
                    # If all moves lead to 'A', we end up 'A'; if we see a 'T', we returned already
            return best_outcome
        else:
            # Aoki's turn
            best_outcome = 'T'  # If we cannot find a winning route for A, we'll remain 'T'
            for pos in range(9):
                if not (usedmask & (1 << pos)):
                    # A picks it
                    new_Amask = Amask | (1 << pos)
                    new_Ascore = Ascore + A[pos]
                    new_usedmask = usedmask | (1 << pos)
                    outcome = dfs(Tmask, new_Amask, Tscore, new_Ascore, new_usedmask, 0)
                    if outcome == 'A':
                        return 'A'  # Aoki can force an A outcome
                    # Otherwise outcome == 'T'
            return best_outcome

    # Initial call
    winner = dfs(0, 0, 0, 0, 0, 0)
    if winner == 'T':
        print("Takahashi")
    else:
        print("Aoki")

# Let's call solve() to match the requirement
if __name__ == "__main__":
    solve()