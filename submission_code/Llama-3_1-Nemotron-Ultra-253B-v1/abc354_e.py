from functools import lru_cache

n = int(input())
cards = [tuple(map(int, input().split())) for _ in range(n)]

@lru_cache(maxsize=None)
def can_win(mask):
    if mask == 0:
        return False
    for i in range(n):
        if not (mask & (1 << i)):
            continue
        for j in range(i + 1, n):
            if not (mask & (1 << j)):
                continue
            a_i, b_i = cards[i]
            a_j, b_j = cards[j]
            if a_i == a_j or b_i == b_j:
                new_mask = mask ^ (1 << i) ^ (1 << j)
                if not can_win(new_mask):
                    return True
    return False

print("Takahashi" if can_win((1 << n) - 1) else "Aoki")