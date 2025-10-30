n = int(input())
cards = []
for _ in range(n):
    a, b = map(int, input().split())
    cards.append((a, b))

memo = {}

def can_win(mask):
    if mask in memo:
        return memo[mask]
    
    available_cards = []
    for i in range(n):
        if mask & (1 << i):
            available_cards.append(i)
    
    if len(available_cards) < 2:
        memo[mask] = False
        return False
    
    for i in range(len(available_cards)):
        for j in range(i + 1, len(available_cards)):
            card_i = available_cards[i]
            card_j = available_cards[j]
            
            # Check if these cards can be removed together
            if (cards[card_i][0] == cards[card_j][0] or 
                cards[card_i][1] == cards[card_j][1]):
                
                new_mask = mask & ~(1 << card_i) & ~(1 << card_j)
                if not can_win(new_mask):
                    memo[mask] = True
                    return True
    
    memo[mask] = False
    return False

initial_mask = (1 << n) - 1

if can_win(initial_mask):
    print("Takahashi")
else:
    print("Aoki")