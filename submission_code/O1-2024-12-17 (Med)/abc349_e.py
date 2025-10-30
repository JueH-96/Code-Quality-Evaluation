def main():
    import sys
    sys.setrecursionlimit(10**7)

    # Read input
    A = []
    for _ in range(3):
        row = list(map(int, sys.stdin.readline().strip().split()))
        A.extend(row)

    # Possible winning triples (0-based indexing of the 3x3 cells):
    #  0 1 2
    #  3 4 5
    #  6 7 8
    wins = [
        (0,1,2), (3,4,5), (6,7,8),  # rows
        (0,3,6), (1,4,7), (2,5,8),  # cols
        (0,4,8), (2,4,6)            # diagonals
    ]

    # Fast check if a bitmask (mask) contains any 3-in-a-row
    def check_win(mask):
        for x,y,z in wins:
            if (mask & (1 << x)) and (mask & (1 << y)) and (mask & (1 << z)):
                return True
        return False

    INF = 10**18

    # We'll implement alpha-beta search from Takahashi's perspective.
    # tmask: bitmask of red cells (Takahashi)
    # amask: bitmask of blue cells (Aoki)
    # diff : (score_T - score_A) so far
    # turn : how many moves have been played so far (0..9)
    #        If turn%2 == 0 => it's Takahashi's move, else Aoki's move.
    # We return the best achievable final (score_T - score_A) from T's point of view,
    # or +INF if T can force immediate 3-in-a-row, or -INF if A forces immediate 3-in-a-row.

    def alpha_beta(tmask, amask, diff, turn, alpha, beta):
        used_count = turn
        # If all cells used, game ends => return final difference
        if used_count == 9:
            return diff

        # Current player
        isTakahashi = (turn % 2 == 0)

        # Try all possible cells
        used_mask = tmask | amask
        if isTakahashi:
            # Takahashi's move: maximize result
            best_val = -INF
            for c in range(9):
                if not (used_mask & (1 << c)):
                    new_tmask = tmask | (1 << c)
                    new_diff = diff + A[c]
                    # Check if this forms a winning 3-in-a-row for T
                    if check_win(new_tmask):
                        # Immediate win => +INF
                        return INF
                    val = alpha_beta(new_tmask, amask, new_diff, turn+1, alpha, beta)
                    if val > best_val:
                        best_val = val
                    if best_val > alpha:
                        alpha = best_val
                    if alpha >= beta:
                        break
            return best_val
        else:
            # Aoki's move: minimize result
            best_val = INF
            for c in range(9):
                if not (used_mask & (1 << c)):
                    new_amask = amask | (1 << c)
                    new_diff = diff - A[c]
                    # Check if this forms a winning 3-in-a-row for A
                    if check_win(new_amask):
                        # Immediate win => -INF
                        return -INF
                    val = alpha_beta(tmask, new_amask, new_diff, turn+1, alpha, beta)
                    if val < best_val:
                        best_val = val
                    if best_val < beta:
                        beta = best_val
                    if alpha >= beta:
                        break
            return best_val

    result = alpha_beta(0, 0, 0, 0, -INF, INF)

    # If returned value is >= 0, Takahashi wins; else Aoki wins
    # (Because sum of all A[i] is odd, final difference cannot be exactly 0
    #  if we reach filling all cells. However, we use >= 0 check to include
    #  the +INF case, etc.)
    if result >= 0:
        print("Takahashi")
    else:
        print("Aoki")

# Do not forget to call main()
if __name__ == "__main__":
    main()