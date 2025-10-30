import sys
from functools import lru_cache

def main():
    N = int(sys.stdin.readline())
    cards = []
    for _ in range(N):
        a, b = map(int, sys.stdin.readline().split())
        cards.append((a, b))

    @lru_cache(maxsize=None)
    def can_win(mask):
        remaining = []
        for i in range(N):
            if mask & (1 << i):
                remaining.append(i)
        # Check all possible pairs
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                a_idx = remaining[i]
                b_idx = remaining[j]
                if cards[a_idx][0] == cards[b_idx][0] or cards[a_idx][1] == cards[b_idx][1]:
                    new_mask = mask & ~((1 << a_idx) | (1 << b_idx))
                    if not can_win(new_mask):
                        return True
        return False

    initial_mask = (1 << N) - 1
    if can_win(initial_mask):
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == "__main__":
    main()