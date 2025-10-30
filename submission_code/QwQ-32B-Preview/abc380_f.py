def main():
    import sys
    from functools import lru_cache

    # Read input
    N, M, L = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    C = list(map(int, sys.stdin.readline().split()))

    # Assign unique ranks to all cards
    all_cards = A + B + C
    sorted_cards = sorted(all_cards)
    rank = {c: i for i, c in enumerate(sorted_cards)}

    # Initialize bit masks for hands and table
    Takahashi_hand = 0
    for a in A:
        Takahashi_hand |= (1 << rank[a])

    Aoki_hand = 0
    for b in B:
        Aoki_hand |= (1 << rank[b])

    table = 0
    for c in C:
        table |= (1 << rank[c])

    # Precompute eligible pickups for each card rank
    eligible_pickups = []
    for r in range(N + M + L):
        mask = 0
        for c in all_cards:
            if rank[c] < r:
                mask |= (1 << rank[c])
        eligible_pickups.append(mask)

    # Recursive function with memoization
    @lru_cache(maxsize=None)
    def can_win(Takahashi_hand, Aoki_hand, table, current_player):
        if current_player == 0:  # Takahashi's turn
            if Takahashi_hand == 0:
                return False
            hand = Takahashi_hand
        else:  # Aoki's turn
            if Aoki_hand == 0:
                return False
            hand = Aoki_hand

        # Iterate through each card in the current player's hand
        while hand:
            card = rank[all_cards[hand.bit_length() - 1]]
            hand &= hand - 1  # Remove the chosen card from hand

            # Play the card to the table
            new_table = table | (1 << card)
            pickups = eligible_pickups[card] & new_table

            # Option 1: don't pick up any card
            if current_player == 0:
                new_Takahashi_hand = Takahashi_hand ^ (1 << card)
                if not can_win(new_Takahashi_hand, Aoki_hand, new_table, 1):
                    return True
            else:
                new_Aoki_hand = Aoki_hand ^ (1 << card)
                if not can_win(Takahashi_hand, new_Aoki_hand, new_table, 0):
                    return True

            # Option 2: pick up each eligible card
            while pickups:
                pickup = rank[all_cards[pickups.bit_length() - 1]]
                pickups &= pickups - 1  # Remove the picked up card from pickups

                # Update table and hand accordingly
                new_table ^= (1 << pickup)
                if current_player == 0:
                    new_Takahashi_hand = (Takahashi_hand ^ (1 << card)) | (1 << pickup)
                    if not can_win(new_Takahashi_hand, Aoki_hand, new_table, 1):
                        return True
                else:
                    new_Aoki_hand = (Aoki_hand ^ (1 << card)) | (1 << pickup)
                    if not can_win(Takahashi_hand, new_Aoki_hand, new_table, 0):
                        return True

        return False

    # Determine the winner
    if can_win(Takahashi_hand, Aoki_hand, table, 0):
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == "__main__":
    main()