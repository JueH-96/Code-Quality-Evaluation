from functools import lru_cache

@lru_cache(maxsize=None)
def can_win(m, move):
    if move > 10:
        return False
    if move % 2 == 1:
        # Vanya's turn
        a = m + 1
        b = m - 1
        if a % 3 == 0:
            return True
        if b % 3 == 0:
            return True
        # Check both possibilities after Vova's moves for a
        a_plus = can_win(a + 1, move + 2)
        a_minus = can_win(a - 1, move + 2)
        if a_plus and a_minus:
            return True
        # Check both possibilities after Vova's moves for b
        b_plus = can_win(b + 1, move + 2)
        b_minus = can_win(b - 1, move + 2)
        if b_plus and b_minus:
            return True
        return False
    else:
        # Vova's turn
        c = m + 1
        d = m - 1
        c_win = can_win(c, move + 1)
        d_win = can_win(d, move + 1)
        return c_win and d_win

t = int(input())
for _ in range(t):
    n = int(input())
    print("First" if can_win(n, 1) else "Second")