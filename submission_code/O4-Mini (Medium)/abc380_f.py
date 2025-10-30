import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1000000)
    from functools import lru_cache

    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    L = int(data[2])
    idx = 3
    A = list(map(int, data[idx:idx+N])); idx += N
    B = list(map(int, data[idx:idx+M])); idx += M
    C = list(map(int, data[idx:idx+L])); idx += L

    # Combine all cards into one list of values.
    vals = A + B + C
    K = len(vals)
    fullmask = (1 << K) - 1

    # Precompute for each card i the bitmask of cards j with vals[j] < vals[i].
    less_mask = [0] * K
    for i in range(K):
        m = 0
        vi = vals[i]
        for j in range(K):
            if vals[j] < vi:
                m |= (1 << j)
        less_mask[i] = m

    # Initial bitmasks
    P0 = 0  # Takahashi's hand
    for i in range(N):
        P0 |= 1 << i
    Q0 = 0  # Aoki's hand
    for i in range(N, N+M):
        Q0 |= 1 << i
    # Table is the rest: fullmask ^ P0 ^ Q0

    @lru_cache(None)
    def can_win(P, Q):
        """
        Return True if the current player (with hand mask P) has a winning strategy
        when opponent has hand mask Q, and table is the rest.
        """
        # If no card to play, lose.
        if P == 0:
            return False

        R = fullmask ^ P ^ Q

        # Try all moves: play i from P
        p = P
        while p:
            i = (p & -p).bit_length() - 1
            p &= p - 1
            # Play card i
            P1 = P ^ (1 << i)
            R1 = R | (1 << i)

            # Option 1: do not take any card back
            # Next turn, opponent has P1, current player's role flips:
            if not can_win(Q, P1):
                return True

            # Option 2: take one smaller card j from R1
            # Candidates are those j in R1 with vals[j] < vals[i]:
            take_cands = less_mask[i] & R1
            t = take_cands
            while t:
                j = (t & -t).bit_length() - 1
                t &= t - 1
                # Take j
                P2 = P1 | (1 << j)
                # R2 = R1 ^ (1 << j)  # implied by masks
                if not can_win(Q, P2):
                    return True

        # No winning move found
        return False

    winner = "Takahashi" if can_win(P0, Q0) else "Aoki"
    print(winner)

if __name__ == "__main__":
    main()