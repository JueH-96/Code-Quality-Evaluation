import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10**7)
    A = []
    for _ in range(3):
        A.extend(list(map(int, sys.stdin.readline().split())))
    # Precompute winning line bitmasks over 9 cells (0..8)
    wins = []
    # rows
    for r in range(3):
        wins.append((1 << (3*r)) | (1 << (3*r+1)) | (1 << (3*r+2)))
    # cols
    for c in range(3):
        wins.append((1 << c) | (1 << (c+3)) | (1 << (c+6)))
    # diags
    wins.append((1<<0)|(1<<4)|(1<<8))
    wins.append((1<<2)|(1<<4)|(1<<6))

    from functools import lru_cache
    INF = 10**18

    @lru_cache(None)
    def dfs(red_mask, blue_mask):
        # whose turn?
        used = red_mask | blue_mask
        moves_done = (used.bit_count())
        tak_turn = (moves_done % 2 == 0)
        # If no moves left, compare scores
        if moves_done == 9:
            # No more moves; zero incremental because we've been accumulating via returns
            return 0
        if tak_turn:
            best = -INF
        else:
            best = INF

        # try all available moves
        for pos in range(9):
            bit = 1 << pos
            if used & bit: continue
            if tak_turn:
                nm = red_mask | bit
                # check Tak win
                for w in wins:
                    if nm & w == w:
                        # immediate Tak win: huge positive
                        best = max(best, INF)
                        break
                else:
                    # no instant win, recurse
                    res = dfs(nm, blue_mask)
                    # add points for Tak's move
                    res += A[pos]
                    best = max(best, res)
            else:
                nm = blue_mask | bit
                # check Aoki win
                for w in wins:
                    if nm & w == w:
                        # immediate Aoki win: huge negative
                        best = min(best, -INF)
                        break
                else:
                    res = dfs(red_mask, nm)
                    # subtract points since Aoki's pick reduces Tak-Aoki difference
                    res -= A[pos]
                    best = min(best, res)
        return best

    val = dfs(0,0)
    # if val>0, Takahashi wins; else Aoki
    if val > 0:
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == "__main__":
    main()