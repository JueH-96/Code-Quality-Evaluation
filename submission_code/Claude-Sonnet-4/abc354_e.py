# YOUR CODE HERE
def solve():
    n = int(input())
    cards = []
    for _ in range(n):
        a, b = map(int, input().split())
        cards.append((a, b))
    
    # Memoization dictionary
    memo = {}
    
    def can_win(mask):
        if mask in memo:
            return memo[mask]
        
        # Get list of remaining cards
        remaining = []
        for i in range(n):
            if mask & (1 << i):
                remaining.append(i)
        
        # Try all pairs of remaining cards
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                card1_idx = remaining[i]
                card2_idx = remaining[j]
                card1 = cards[card1_idx]
                card2 = cards[card2_idx]
                
                # Check if we can remove this pair
                if card1[0] == card2[0] or card1[1] == card2[1]:
                    # Remove these two cards and check if opponent loses
                    new_mask = mask & ~(1 << card1_idx) & ~(1 << card2_idx)
                    if not can_win(new_mask):
                        memo[mask] = True
                        return True
        
        # If no winning move found, current player loses
        memo[mask] = False
        return False
    
    # Initial state: all cards are on the table
    initial_mask = (1 << n) - 1
    
    if can_win(initial_mask):
        print("Takahashi")
    else:
        print("Aoki")

solve()