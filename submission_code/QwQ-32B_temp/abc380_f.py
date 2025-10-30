import sys
from functools import lru_cache

def main():
    sys.setrecursionlimit(1 << 25)
    N, M, L = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    C = list(map(int, sys.stdin.readline().split()))

    A.sort()
    B.sort()
    C.sort()

    initial_takahashi = tuple(A)
    initial_aoki = tuple(B)
    initial_table = tuple(C)

    @lru_cache(maxsize=None)
    def can_win(takahashi_hand, aoki_hand, table, turn):
        if turn == 0:  # Takahashi's turn
            if not takahashi_hand:
                return False
            for i in range(len(takahashi_hand)):
                card = takahashi_hand[i]
                new_th = list(takahashi_hand)
                del new_th[i]
                new_th_tuple = tuple(new_th)
                new_table = list(table) + [card]
                new_table.sort()
                new_table_tuple = tuple(new_table)
                available = [c for c in new_table if c < card]
                # Option 1: take none
                next_state = (new_th_tuple, aoki_hand, new_table_tuple, 1)
                if not can_win(*next_state):
                    return True
                # Option 2: take any of available
                for take in available:
                    new_table_after = list(new_table)
                    new_table_after.remove(take)
                    new_table_after.sort()
                    new_table_after_tuple = tuple(new_table_after)
                    new_th_after = list(new_th) + [take]
                    new_th_after.sort()
                    new_th_after_tuple = tuple(new_th_after)
                    next_state_take = (new_th_after_tuple, aoki_hand, new_table_after_tuple, 1)
                    if not can_win(*next_state_take):
                        return True
            return False
        else:  # Aoki's turn
            if not aoki_hand:
                return False
            for i in range(len(aoki_hand)):
                card = aoki_hand[i]
                new_ah = list(aoki_hand)
                del new_ah[i]
                new_ah_tuple = tuple(new_ah)
                new_table = list(table) + [card]
                new_table.sort()
                new_table_tuple = tuple(new_table)
                available = [c for c in new_table if c < card]
                # Option 1: take none
                next_state = (takahashi_hand, new_ah_tuple, new_table_tuple, 0)
                if not can_win(*next_state):
                    return True
                # Option 2: take any of available
                for take in available:
                    new_table_after = list(new_table)
                    new_table_after.remove(take)
                    new_table_after.sort()
                    new_table_after_tuple = tuple(new_table_after)
                    new_ah_after = list(new_ah) + [take]
                    new_ah_after.sort()
                    new_ah_after_tuple = tuple(new_ah_after)
                    next_state_take = (takahashi_hand, new_ah_after_tuple, new_table_after_tuple, 0)
                    if not can_win(*next_state_take):
                        return True
            return False

    result = can_win(initial_takahashi, initial_aoki, initial_table, 0)
    print("Takahashi" if result else "Aoki")

if __name__ == "__main__":
    main()