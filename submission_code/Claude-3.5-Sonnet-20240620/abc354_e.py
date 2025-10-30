# YOUR CODE HERE
from collections import defaultdict

def can_win(cards):
    if not cards:
        return False
    
    front = defaultdict(list)
    back = defaultdict(list)
    
    for i, (a, b) in enumerate(cards):
        front[a].append(i)
        back[b].append(i)
    
    for d in [front, back]:
        for num in d:
            if len(d[num]) >= 2:
                new_cards = [card for i, card in enumerate(cards) if i not in d[num][:2]]
                if not can_win(new_cards):
                    return True
    
    return False

N = int(input())
cards = [tuple(map(int, input().split())) for _ in range(N)]

print("Takahashi" if can_win(cards) else "Aoki")