import sys
from functools import lru_cache

def main() -> None:
    sys.setrecursionlimit(1 << 25)

    # -------------------- read input --------------------
    N, M, L = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))          # Takahashi's cards
    B = list(map(int, sys.stdin.readline().split()))          # Aoki's cards
    C = list(map(int, sys.stdin.readline().split()))          # cards on the table

    values = A + B + C                    # value of every card, index = 0 .. total-1
    total = len(values)                   # = N + M + L
    FULL = (1 << total) - 1

    # bitmask of cards that are in each player's hand at the initial position
    T_start = (1 << N) - 1                           # first  N indices
    A_start = ((1 << M) - 1) << N                    # next   M indices
    # cards on the table are the remaining bits

    # ----------------------------------------------------
    def bits(mask: int):
        """generator: indices of the bits set in mask"""
        while mask:
            lsb = mask & -mask
            idx = (lsb.bit_length() - 1)
            yield idx
            mask ^= lsb

    # ----------------------------------------------------
    @lru_cache(maxsize=None)
    def takahashi_win(T_mask: int, A_mask: int, turn: int) -> bool:
        """
        return True  if the player who has the move in this position wins,
        return False if he loses (assuming optimal play).
        turn : 0 -> Takahashi to move, 1 -> Aoki to move
        """
        # check if current player has a card to play
        if turn == 0:          # Takahashi's move
            current_hand = T_mask
        else:                  # Aoki's move
            current_hand = A_mask

        if current_hand == 0:
            # cannot move -> current player loses
            return False

        # for every possible card to play
        for card in bits(current_hand):
            val_card = values[card]

            # remove card from current player's hand (it goes to the table)
            if turn == 0:
                new_T = T_mask & ~(1 << card)
                new_A = A_mask
            else:
                new_T = T_mask
                new_A = A_mask & ~(1 << card)

            # table after playing the card (card is now on the table)
            table_mask = FULL ^ new_T ^ new_A         # complement of hands

            # list of possible picks: one card with a smaller value, or pick nothing
            smaller_cards = [idx for idx in bits(table_mask)
                             if values[idx] < val_card]

            # "pick nothing" option
            next_T = new_T
            next_A = new_A
            if not takahashi_win(next_T, next_A, 1 - turn):
                return True        # opponent is in a losing position

            # try picking each smaller card
            for picked in smaller_cards:
                if turn == 0:
                    next_T = new_T | (1 << picked)
                    next_A = new_A
                else:
                    next_T = new_T
                    next_A = new_A | (1 << picked)

                if not takahashi_win(next_T, next_A, 1 - turn):
                    return True

        # none of the moves gives a win -> current player loses
        return False

    winner = "Takahashi" if takahashi_win(T_start, A_start, 0) else "Aoki"
    print(winner)


if __name__ == "__main__":
    main()