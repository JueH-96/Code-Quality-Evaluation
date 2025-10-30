from itertools import combinations
from functools import lru_cache

def main():
    N = int(input())
    cards = [tuple(map(int, input().split())) for _ in range(N)]
    
    # Precompute all possible pairs that can be removed
    pairs = []
    for i in range(N):
        for j in range(i+1, N):
            if cards[i][0] == cards[j][0] or cards[i][1] == cards[j][1]:
                pairs.append((i, j))
    
    # Use a bitmask to represent the state of the cards
    # 1 means the card is still on the table, 0 means it's removed
    # Initially, all cards are on the table
    initial_mask = (1 << N) - 1
    
    @lru_cache(maxsize=None)
    def can_win(mask):
        # Iterate over all possible pairs
        for i, j in pairs:
            if (mask & (1 << i)) and (mask & (1 << j)):
                # Remove the pair and see if the opponent can win
                new_mask = mask ^ (1 << i) ^ (1 << j)
                if not can_win(new_mask):
                    return True
        return False
    
    if can_win(initial_mask):
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == "__main__":
    main()