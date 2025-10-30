def can_remove_pair(cards, i, j):
    """Check if cards i and j can be removed (share front or back number)"""
    return cards[i][0] == cards[j][0] or cards[i][1] == cards[j][1]

def solve(n, cards):
    # Memoization for game states
    # State is represented as a bitmask where bit i is 1 if card i is still on table
    memo = {}
    
    def is_winning(mask):
        """Returns True if current player can win from this state"""
        if mask in memo:
            return memo[mask]
        
        # Try all possible pairs of remaining cards
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            for j in range(i + 1, n):
                if not (mask & (1 << j)):
                    continue
                
                # Check if we can remove this pair
                if can_remove_pair(cards, i, j):
                    # Remove cards i and j
                    new_mask = mask & ~(1 << i) & ~(1 << j)
                    
                    # If opponent loses from the new state, we win
                    if not is_winning(new_mask):
                        memo[mask] = True
                        return True
        
        # No winning move found
        memo[mask] = False
        return False
    
    # Initial state: all cards are on the table
    initial_mask = (1 << n) - 1
    
    if is_winning(initial_mask):
        return "Takahashi"
    else:
        return "Aoki"

# Read input
n = int(input())
cards = []
for _ in range(n):
    a, b = map(int, input().split())
    cards.append((a, b))

# Solve and print result
print(solve(n, cards))