import sys
sys.setrecursionlimit(10**7)

def main():
    # Read inputs
    data = sys.stdin.read().strip().split()
    N, M, L = map(int, data[:3])
    A = list(map(int, data[3:3+N]))
    B = list(map(int, data[3+N:3+N+M]))
    C = list(map(int, data[3+N+M:]))

    # Combine all card values in a single list
    vals = A + B + C
    total_cards = N + M + L

    # Initial bitmasks for each player's hand
    # We'll assign indices 0..(N-1) to Takahashi's initial cards,
    # indices N..(N+M-1) to Aoki's initial cards,
    # indices (N+M)..(N+M+L-1) to table cards.
    # But note that as play proceeds, cards can move around arbitrarily.
    # We'll track who has which cards via bitmasks tMask, aMask,
    # and deduce tableMask = fullMask ^ (tMask|aMask).
    
    tMask = 0
    for i in range(N):
        tMask |= (1 << i)
    
    aMask = 0
    for i in range(M):
        aMask |= (1 << (N + i))
    
    # We won't store tableMask explicitly in the state,
    # because tableMask = fullMask ^ (tMask | aMask).
    fullMask = (1 << total_cards) - 1

    # Memo dictionary: (tMask, aMask, turn) -> bool
    # True means the current player (turn 0=T, turn 1=A) has a winning strategy,
    # False means the current player will lose if both play optimally.
    memo = {}

    def can_win(tMask, aMask, turn):
        """Return True if the current player (turn) is winning from state (tMask, aMask)."""
        state_key = (tMask, aMask, turn)
        if state_key in memo:
            return memo[state_key]

        # If it's Takahashi's turn and he has no cards -> he cannot move -> he loses
        if turn == 0:
            if tMask == 0:
                memo[state_key] = False
                return False
        else:
            # Aoki's turn, no cards -> loses
            if aMask == 0:
                memo[state_key] = False
                return False

        tableMask = fullMask ^ (tMask | aMask)

        if turn == 0:
            # Try all moves for Takahashi
            # For each card c in T's hand
            for c in range(total_cards):
                if (tMask & (1 << c)) != 0:
                    # Place card c on the table
                    newTMask = tMask ^ (1 << c)
                    newTableMask = tableMask | (1 << c)

                    # Option 1: Do not pick up any card
                    if not can_win(newTMask, aMask, 1):
                        memo[state_key] = True
                        return True

                    # Option 2: Pick up exactly one smaller card
                    # We look for cards in newTableMask that are strictly smaller than vals[c]
                    for d in range(total_cards):
                        if (newTableMask & (1 << d)) != 0 and vals[d] < vals[c]:
                            # Pick up d from the table
                            nxtTMask = newTMask | (1 << d)
                            nxtTableMask = newTableMask ^ (1 << d)  # remove from table
                            # Now see if this leads the opponent to a losing state
                            if not can_win(nxtTMask, aMask, 1):
                                memo[state_key] = True
                                return True
            # If no move leads to opponent losing, current player loses
            memo[state_key] = False
            return False
        else:
            # Aoki's turn => symmetrical
            for c in range(total_cards):
                if (aMask & (1 << c)) != 0:
                    # Place card c on the table
                    newAMask = aMask ^ (1 << c)
                    newTableMask = tableMask | (1 << c)

                    # Option 1: Do not pick up any card
                    if not can_win(tMask, newAMask, 0):
                        memo[state_key] = True
                        return True

                    # Option 2: Pick up exactly one smaller card
                    for d in range(total_cards):
                        if (newTableMask & (1 << d)) != 0 and vals[d] < vals[c]:
                            nxtAMask = newAMask | (1 << d)
                            nxtTableMask = newTableMask ^ (1 << d)
                            if not can_win(tMask, nxtAMask, 0):
                                memo[state_key] = True
                                return True
            memo[state_key] = False
            return False

    # Determine the outcome from the initial state
    if can_win(tMask, aMask, 0):
        print("Takahashi")
    else:
        print("Aoki")

# Don't forget to call main()
if __name__ == "__main__":
    main()