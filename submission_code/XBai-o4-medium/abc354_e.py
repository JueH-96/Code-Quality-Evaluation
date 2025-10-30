import sys
from functools import lru_cache

def main():
    n = int(sys.stdin.readline())
    cards = [tuple(map(int, sys.stdin.readline().split())) for _ in range(n)]

    @lru_cache(maxsize=None)
    def can_win(mask):
        count = bin(mask).count('1')
        if count < 2:
            return False  # No possible moves

        # Iterate over all possible pairs (i, j)
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            for j in range(i + 1, n):
                if not (mask & (1 << j)):
                    continue
                a_i, b_i = cards[i]
                a_j, b_j = cards[j]
                if a_i == a_j or b_i == b_j:
                    new_mask = mask ^ ((1 << i) | (1 << j))
                    if not can_win(new_mask):
                        return True
        return False  # All moves lead to opponent winning

    full_mask = (1 << n) - 1
    if can_win(full_mask):
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == "__main__":
    main()