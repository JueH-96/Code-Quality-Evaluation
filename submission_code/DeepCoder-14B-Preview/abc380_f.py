import sys
from functools import lru_cache

def main():
    N, M, L = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    C = list(map(int, sys.stdin.readline().split()))

    T_hand = tuple(sorted(A))
    A_hand = tuple(sorted(B))
    table = tuple(sorted(C))

    @lru_cache(maxsize=None)
    def game(T_hand, A_hand, table, is_takahashi_turn):
        if is_takahashi_turn:
            current_hand = T_hand
        else:
            current_hand = A_hand

        if not current_hand:
            return False  # Current player cannot move, loses

        for i in range(len(current_hand)):
            x = current_hand[i]

            # Create new_hand by removing the i-th element
            new_hand = current_hand[:i] + current_hand[i+1:]

            # Add the played card to the table
            new_table_list = list(table)
            new_table_list.append(x)
            new_table_sorted = tuple(sorted(new_table_list))

            # Collect all possible cards to take (s < x)
            S = [s for s in new_table_sorted if s < x]

            # Option 1: Do not take any card
            if is_takahashi_turn:
                next_T_hand = new_hand
                next_A_hand = A_hand
                next_table = new_table_sorted
                next_turn = False
                if not game(next_T_hand, next_A_hand, next_table, next_turn):
                    return True
            else:
                next_T_hand = T_hand
                next_A_hand = new_hand
                next_table = new_table_sorted
                next_turn = True
                if not game(next_T_hand, next_A_hand, next_table, next_turn):
                    return True

            # Option 2: Take each possible card s in S
            for s in S:
                # Remove s from the new table
                temp_table = list(new_table_sorted)
                try:
                    temp_table.remove(s)
                except ValueError:
                    continue  # This should not happen as s is in S
                new_table_take = tuple(temp_table)

                if is_takahashi_turn:
                    # Add s to Takahashi's hand
                    new_T_hand = new_hand + (s,)
                    new_A_hand = A_hand
                    if not game(new_T_hand, new_A_hand, new_table_take, False):
                        return True
                else:
                    # Add s to Aoki's hand
                    new_A_hand = new_hand + (s,)
                    new_T_hand = T_hand
                    if not game(new_T_hand, new_A_hand, new_table_take, True):
                        return True

        return False  # No winning moves found

    result = game(T_hand, A_hand, table, True)
    if result:
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == '__main__':
    main()