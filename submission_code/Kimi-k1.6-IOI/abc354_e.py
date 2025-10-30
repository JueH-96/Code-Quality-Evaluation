import sys
from functools import lru_cache

def main():
    N = int(sys.stdin.readline())
    cards = []
    for _ in range(N):
        a, b = map(int, sys.stdin.readline().split())
        cards.append((a, b))
    
    # Precompute valid pairs
    valid = [[False]*N for _ in range(N)]
    for i in range(N):
        a1, b1 = cards[i]
        for j in range(i+1, N):
            a2, b2 = cards[j]
            if a1 == a2 or b1 == b2:
                valid[i][j] = True
    
    @lru_cache(maxsize=None)
    def dp(mask):
        for i in range(N):
            if not (mask & (1 << i)):
                continue
            for j in range(i+1, N):
                if not (mask & (1 << j)):
                    continue
                if valid[i][j]:
                    new_mask = mask ^ ((1 << i) | (1 << j))
                    if not dp(new_mask):
                        return True
        return False
    
    initial_mask = (1 << N) - 1
    print("Takahashi" if dp(initial_mask) else "Aoki")

if __name__ == '__main__':
    main()