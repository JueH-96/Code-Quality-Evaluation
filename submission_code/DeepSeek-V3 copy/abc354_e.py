from itertools import combinations
from functools import lru_cache

def main():
    N = int(input())
    cards = [tuple(map(int, input().split())) for _ in range(N)]
    
    @lru_cache(maxsize=None)
    def can_win(mask):
        for i in range(N):
            if not (mask & (1 << i)):
                for j in range(i+1, N):
                    if not (mask & (1 << j)):
                        if cards[i][0] == cards[j][0] or cards[i][1] == cards[j][1]:
                            new_mask = mask | (1 << i) | (1 << j)
                            if not can_win(new_mask):
                                return True
        return False
    
    if can_win(0):
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == "__main__":
    main()