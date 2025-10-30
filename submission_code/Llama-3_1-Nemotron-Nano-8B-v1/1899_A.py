from functools import lru_cache

@lru_cache(maxsize=None)
def can_win(num, moves, is_vanya_turn):
    if moves == 10:
        return False
    if is_vanya_turn:
        for delta in (1, -1):
            next_num = num + delta
            if next_num % 3 == 0:
                return True
            if can_win(next_num, moves + 1, False):
                return True
        return False
    else:
        for delta in (1, -1):
            next_num = num + delta
            if not can_win(next_num, moves + 1, True):
                return False
        return True

t = int(input())
for _ in range(t):
    n = int(input())
    if can_win(n, 0, True):
        print("First")
    else:
        print("Second")