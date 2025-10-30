import sys
from functools import lru_cache

def main():
    n = int(sys.stdin.readline())
    cards = []
    for _ in range(n):
        a, b = map(int, sys.stdin.readline().split())
        cards.append((a, b))
    
    @lru_cache(maxsize=None)
    def can_win(mask):
        # Check all possible pairs of cards present in the mask
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
        return False
    
    initial_mask = (1 << n) - 1
    result = can_win(initial_mask)
    print("Takahashi" if result else "Aoki")

if __name__ == "__main__":
    main()