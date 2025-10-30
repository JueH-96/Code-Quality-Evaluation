# YOUR CODE HERE
n = int(input())
cards = []
for _ in range(n):
    a, b = map(int, input().split())
    cards.append((a, b))

# Memoization for game states
# dp[mask] = True if the current player can win from this state
dp = {}

def can_win(mask):
    if mask in dp:
        return dp[mask]
    
    # Try all possible pairs
    for i in range(n):
        if not (mask & (1 << i)):
            continue
        for j in range(i + 1, n):
            if not (mask & (1 << j)):
                continue
            
            # Check if we can remove this pair
            if cards[i][0] == cards[j][0] or cards[i][1] == cards[j][1]:
                # Remove the pair and check if opponent loses
                new_mask = mask & ~(1 << i) & ~(1 << j)
                if not can_win(new_mask):
                    dp[mask] = True
                    return True
    
    dp[mask] = False
    return False

# Initial state: all cards are on the table
initial_mask = (1 << n) - 1

# If Takahashi (first player) can win from the initial state
if can_win(initial_mask):
    print("Takahashi")
else:
    print("Aoki")