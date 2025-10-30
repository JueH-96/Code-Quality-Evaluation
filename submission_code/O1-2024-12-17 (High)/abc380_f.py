def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N, M, L = map(int, input_data[:3])
    idx = 3
    A_cards = list(map(int, input_data[idx:idx+N]))
    idx += N
    B_cards = list(map(int, input_data[idx:idx+M]))
    idx += M
    C_cards = list(map(int, input_data[idx:idx+L]))

    # All cards in one list for extracting distinct values
    all_cards = A_cards + B_cards + C_cards
    distinct_vals = sorted(set(all_cards))
    D = len(distinct_vals)

    # Map each distinct value to its index
    val_to_idx = {}
    for i, v in enumerate(distinct_vals):
        val_to_idx[v] = i

    # Count how many of each distinct value Takahashi, Aoki, and the table initially hold
    Tinit = [0]*D
    Ainit = [0]*D
    for v in A_cards:
        Tinit[val_to_idx[v]] += 1
    for v in B_cards:
        Ainit[val_to_idx[v]] += 1

    # totalCount[i] = total number of cards with value distinct_vals[i]
    totalCount = [0]*D
    for v in all_cards:
        totalCount[val_to_idx[v]] += 1

    # Precompute which indices are strictly smaller than i (since vals is strictly ascending)
    smaller = [[] for _ in range(D)]
    for i in range(D):
        # all j < i implies distinct_vals[j] < distinct_vals[i]
        # so smaller[i] = [0..i-1]
        smaller[i] = list(range(i))

    from functools import lru_cache

    @lru_cache(None)
    def rec(t_tuple, a_tuple, turn):
        """
        t_tuple, a_tuple are tuples of length D indicating how many copies
        of each distinct value Takahashi or Aoki holds.
        turn = True means it's Takahashi's turn, False means Aoki's turn.
        Returns True if the current 'turn' player has a winning strategy from this state.
        """
        t = list(t_tuple)
        a = list(a_tuple)
        sumT = sum(t)
        sumA = sum(a)

        # If it's Takahashi's turn and he has no cards, he loses immediately.
        if turn and sumT == 0:
            return False
        # If it's Aoki's turn and he has no cards, he loses immediately.
        if (not turn) and sumA == 0:
            return False

        # Compute how many of each value are on the table
        table = [totalCount[i] - t[i] - a[i] for i in range(D)]

        if turn:
            # Takahashi's turn
            for i in range(D):
                if t[i] > 0:
                    # Play one card of value i
                    t[i] -= 1
                    table[i] += 1

                    # Option 1: Don't pick up any smaller card
                    if not rec(tuple(t), tuple(a), False):
                        # Revert and return True (found a winning move)
                        t[i] += 1
                        table[i] -= 1
                        return True

                    # Option 2: Pick up exactly one smaller card j (if any available)
                    for j in smaller[i]:
                        if table[j] > 0:
                            t[j] += 1
                            table[j] -= 1
                            if not rec(tuple(t), tuple(a), False):
                                # Revert fully and return True
                                t[j] -= 1
                                table[j] += 1
                                t[i] += 1
                                table[i] -= 1
                                return True
                            # Revert picking up j
                            t[j] -= 1
                            table[j] += 1

                    # Revert playing i
                    t[i] += 1
                    table[i] -= 1

            # If no move leads to opponent losing, this is a losing state
            return False
        else:
            # Aoki's turn (same logic, but we manipulate 'a')
            for i in range(D):
                if a[i] > 0:
                    # Play one card of value i
                    a[i] -= 1
                    table[i] += 1

                    # Option 1: Don't pick up anything
                    if not rec(tuple(t), tuple(a), True):
                        a[i] += 1
                        table[i] -= 1
                        return True

                    # Option 2: Pick up exactly one smaller card j
                    for j in smaller[i]:
                        if table[j] > 0:
                            a[j] += 1
                            table[j] -= 1
                            if not rec(tuple(t), tuple(a), True):
                                a[j] -= 1
                                table[j] += 1
                                a[i] += 1
                                table[i] -= 1
                                return True
                            a[j] -= 1
                            table[j] += 1

                    # Revert
                    a[i] += 1
                    table[i] -= 1

            return False

    # Check if Takahashi (True turn) wins from the initial configuration
    if rec(tuple(Tinit), tuple(Ainit), True):
        print("Takahashi")
    else:
        print("Aoki")

# Don't forget to call main
if __name__ == "__main__":
    main()