import sys
from functools import lru_cache

def main():
    N, M, L = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    C = list(map(int, sys.stdin.readline().split()))

    A_tuple = tuple(sorted(A))
    B_tuple = tuple(sorted(B))
    C_tuple = tuple(sorted(C))

    @lru_cache(maxsize=None)
    def can_win(a, b, c, is_takahashi_turn):
        current_hand = a if is_takahashi_turn else b
        opponent_hand = b if is_takahashi_turn else a

        if not current_hand:
            return False

        for x in current_hand:
            new_a = list(a)
            new_a.remove(x)
            new_a = tuple(sorted(new_a))
            new_c = list(c) + [x]
            new_c = tuple(sorted(new_c))

            if not can_win(new_a, opponent_hand, new_c, not is_takahashi_turn):
                return True

            possible_ys = [y for y in c if y < x]
            for y in possible_ys:
                if is_takahashi_turn:
                    new_a_y = list(a)
                    new_a_y.remove(x)
                    new_a_y.append(y)
                    new_a_y = tuple(sorted(new_a_y))
                    new_b_y = opponent_hand
                    new_c_y = list(c)
                    new_c_y.remove(y)
                    new_c_y.append(x)
                    new_c_y = tuple(sorted(new_c_y))
                else:
                    new_b_y = list(b)
                    new_b_y.remove(x)
                    new_b_y.append(y)
                    new_b_y = tuple(sorted(new_b_y))
                    new_a_y = a
                    new_c_y = list(c)
                    new_c_y.remove(y)
                    new_c_y.append(x)
                    new_c_y = tuple(sorted(new_c_y))

                if not can_win(new_a_y, new_b_y, new_c_y, not is_takahashi_turn):
                    return True

        return False

    result = can_win(A_tuple, B_tuple, C_tuple, True)
    print("Takahashi" if result else "Aoki")

if __name__ == "__main__":
    main()