import sys
from collections import defaultdict

def solve():
    N = int(input())
    cards = [tuple(map(int, input().split())) for _ in range(N)]
    card_map = defaultdict(list)
    for i, (a, b) in enumerate(cards):
        card_map[a].append(i)
        card_map[b].append(i)
    
    def can_remove(cards, mask):
        for a, b in card_map.items():
            if len(b) >= 2:
                for i in range(len(b)):
                    for j in range(i + 1, len(b)):
                        if (mask & (1 << b[i])) and (mask & (1 << b[j])):
                            return True
        return False
    
    def dfs(mask):
        if mask == (1 << N) - 1:
            return False
        for i in range(N):
            if not (mask & (1 << i)):
                new_mask = mask | (1 << i)
                for j in range(i + 1, N):
                    if not (new_mask & (1 << j)):
                        new_mask_j = new_mask | (1 << j)
                        if (cards[i][0] == cards[j][0] or cards[i][1] == cards[j][1]) and not dfs(new_mask_j):
                            return True
                break
        return False
    
    if dfs(0):
        print("Takahashi")
    else:
        print("Aoki")

solve()