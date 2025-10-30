import sys
from functools import lru_cache

def main():
    N = int(sys.stdin.readline())
    cards = []
    for _ in range(N):
        a, b = map(int, sys.stdin.readline().split())
        cards.append((a, b))
    
    # Precompute all valid pairs (i, j) where i < j
    valid_pairs = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if cards[i][0] == cards[j][0] or cards[i][1] == cards[j][1]:
                valid_pairs[i].append(j)
    
    @lru_cache(maxsize=None)
    def can_win(mask):
        for i in range(N):
            if not (mask & (1 << i)):
                continue
            for j in valid_pairs[i]:
                if mask & (1 << j):
                    new_mask = mask ^ (1 << i) ^ (1 << j)
                    if not can_win(new_mask):
                        return True
        return False
    
    initial_mask = (1 << N) - 1
    result = can_win(initial_mask)
    print("Takahashi" if result else "Aoki")

if __name__ == "__main__":
    main()