import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
cards = [(int(data[2*i+1]), int(data[2*i+2])) for i in range(N)]

def can_remove(cards):
    for i in range(len(cards)):
        for j in range(i+1, len(cards)):
            if cards[i][0] == cards[j][0] or cards[i][1] == cards[j][1]:
                return True
    return False

def dfs(cards, turn):
    if not can_remove(cards):
        return turn == 'Takahashi'
    for i in range(len(cards)):
        for j in range(i+1, len(cards)):
            if cards[i][0] == cards[j][0] or cards[i][1] == cards[j][1]:
                new_cards = cards[:i] + cards[i+1:j] + cards[j+1:]
                if dfs(new_cards, 'Aoki' if turn == 'Takahashi' else 'Takahashi'):
                    return turn == 'Aoki'
    return turn == 'Takahashi'

if dfs(cards, 'Takahashi'):
    print('Takahashi')
else:
    print('Aoki')