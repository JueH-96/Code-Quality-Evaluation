def can_remove(cards):
    n = len(cards)
    for i in range(n):
        for j in range(i+1, n):
            if cards[i][0] == cards[j][0] or cards[i][1] == cards[j][1]:
                return True
    return False

def get_next_states(cards):
    n = len(cards)
    states = []
    for i in range(n):
        for j in range(i+1, n):
            if cards[i][0] == cards[j][0] or cards[i][1] == cards[j][1]:
                next_cards = cards[:i] + cards[i+1:j] + cards[j+1:]
                states.append(next_cards)
    return states

def solve(cards, memo={}):
    state = tuple(tuple(card) for card in cards)
    if state in memo:
        return memo[state]
    
    if not can_remove(cards):
        memo[state] = False
        return False
        
    next_states = get_next_states(cards)
    for next_state in next_states:
        if not solve(next_state, memo):
            memo[state] = True
            return True
            
    memo[state] = False
    return False

N = int(input())
cards = []
for _ in range(N):
    a, b = map(int, input().split())
    cards.append([a, b])

if solve(cards):
    print("Takahashi")
else:
    print("Aoki")