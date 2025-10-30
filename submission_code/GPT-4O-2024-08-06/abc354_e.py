def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    cards = [(int(data[2*i+1]), int(data[2*i+2])) for i in range(N)]
    
    from functools import lru_cache
    
    @lru_cache(None)
    def can_win(mask):
        # Check all pairs of cards
        for i in range(N):
            if not (mask & (1 << i)):
                continue
            for j in range(i + 1, N):
                if not (mask & (1 << j)):
                    continue
                # Check if they can be paired
                if cards[i][0] == cards[j][0] or cards[i][1] == cards[j][1]:
                    # Try removing this pair and see if the opponent loses
                    new_mask = mask & ~(1 << i) & ~(1 << j)
                    if not can_win(new_mask):
                        return True
        return False
    
    # Initial state: all cards are on the table
    initial_mask = (1 << N) - 1
    
    if can_win(initial_mask):
        print("Takahashi")
    else:
        print("Aoki")