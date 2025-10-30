def can_remove_pair(cards):
    n = len(cards)
    for i in range(n):
        for j in range(i + 1, n):
            if cards[i][0] == cards[j][0] or cards[i][1] == cards[j][1]:
                return True
    return False

def optimal_play(cards, turn):
    if not can_remove_pair(cards):
        return turn == 1  # If no pairs can be removed, the current player loses
    
    n = len(cards)
    for i in range(n):
        for j in range(i + 1, n):
            if cards[i][0] == cards[j][0] or cards[i][1] == cards[j][1]:
                new_cards = cards[:i] + cards[i+1:j] + cards[j+1:]
                if not optimal_play(new_cards, 1 - turn):
                    return True
    return False

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
cards = [(int(data[2*i+1]), int(data[2*i+2])) for i in range(N)]

if optimal_play(cards, 0):
    print("Takahashi")
else:
    print("Aoki")