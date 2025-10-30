def solve():
    n, m, l = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    c = list(map(int, input().split()))

    memo = {}

    def can_win(t_hand, a_hand, table, turn):
        state = (tuple(sorted(t_hand)), tuple(sorted(a_hand)), tuple(sorted(table)), turn)
        if state in memo:
            return memo[state]

        if turn == 0:  # Takahashi's turn
            if not t_hand:
                memo[state] = False
                return False
            for t_card in sorted(list(t_hand)):
                remaining_t_hand = frozenset(t_hand) - {t_card}
                new_table = tuple(sorted(table + [t_card]))
                smaller_table_cards = sorted([tc for tc in table if tc < t_card])

                # Case 1: Takahashi does not take any card
                if not can_win(frozenset(a_hand), remaining_t_hand, new_table, 1):
                    memo[state] = True
                    return True

                # Case 2: Takahashi takes one card
                for taken_card in smaller_table_cards:
                    new_t_hand_taken = frozenset(remaining_t_hand | {taken_card})
                    new_table_taken = tuple(sorted([tc for tc in table if tc != taken_card] + [t_card]))
                    if not can_win(frozenset(a_hand), new_t_hand_taken, new_table_taken, 1):
                        memo[state] = True
                        return True
            memo[state] = False
            return False
        else:  # Aoki's turn
            if not a_hand:
                memo[state] = False
                return False
            for a_card in sorted(list(a_hand)):
                remaining_a_hand = frozenset(a_hand) - {a_card}
                new_table = tuple(sorted(table + [a_card]))
                smaller_table_cards = sorted([tc for tc in table if tc < a_card])

                # Case 1: Aoki does not take any card
                if not can_win(frozenset(t_hand), remaining_a_hand, new_table, 0):
                    memo[state] = True
                    return True

                # Case 2: Aoki takes one card
                for taken_card in smaller_table_cards:
                    new_a_hand_taken = frozenset(remaining_a_hand | {taken_card})
                    new_table_taken = tuple(sorted([tc for tc in table if tc != taken_card] + [a_card]))
                    if not can_win(frozenset(t_hand), new_a_hand_taken, new_table_taken, 0):
                        memo[state] = True
                        return True
            memo[state] = False
            return False

    if can_win(frozenset(a), frozenset(b), tuple(sorted(c)), 0):
        print("Takahashi")
    else:
        print("Aoki")

solve()