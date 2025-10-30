import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1000000)
    from functools import lru_cache

    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    L = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(M)]
    C = [int(next(it)) for _ in range(L)]
    V = A + B + C
    K = len(V)
    all_mask = (1 << K) - 1

    # Precompute for each card x the bitmask of cards y with V[y] < V[x]
    Ymask = [0] * K
    for x in range(K):
        m = 0
        vx = V[x]
        for y in range(K):
            if V[y] < vx:
                m |= 1 << y
        Ymask[x] = m

    @lru_cache(None)
    def dfs(maskT, maskA, player):
        # If current player has no card, they lose
        if player == 0:
            if maskT == 0:
                return False
        else:
            if maskA == 0:
                return False

        table_mask = all_mask ^ (maskT | maskA)
        # Iterate over choices x in current player's hand
        hand_mask = maskT if player == 0 else maskA
        # For each card x to play
        hm = hand_mask
        while hm:
            lsb = hm & -hm
            x = lsb.bit_length() - 1
            hm ^= lsb
            bitX = 1 << x
            # Remove x from hand, put to table
            if player == 0:
                newT = maskT ^ bitX
                newA = maskA
            else:
                newT = maskT
                newA = maskA ^ bitX
            # Option 1: do not take any card back
            if not dfs(newT, newA, 1 - player):
                return True
            # Option 2: take one card y from old table with V[y] < V[x]
            elig = table_mask & Ymask[x]
            ym = elig
            while ym:
                ybit = ym & -ym
                y = ybit.bit_length() - 1
                ym ^= ybit
                if player == 0:
                    nt = newT | ybit
                    na = newA
                else:
                    nt = newT
                    na = newA | ybit
                if not dfs(nt, na, 1 - player):
                    return True
        # No winning move found
        return False

    # Initial masks
    maskT0 = (1 << N) - 1
    maskA0 = ((1 << M) - 1) << N
    # Table is the rest, no need to set explicitly
    winner = dfs(maskT0, maskA0, 0)
    print("Takahashi" if winner else "Aoki")

# Call main
if __name__ == "__main__":
    main()