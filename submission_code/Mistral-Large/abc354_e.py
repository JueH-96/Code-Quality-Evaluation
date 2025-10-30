import sys
from collections import defaultdict

def can_win(cards, turn):
    if not cards:
        return turn == 0

    front = defaultdict(list)
    back = defaultdict(list)

    for i, (a, b) in enumerate(cards):
        front[a].append(i)
        back[b].append(i)

    for key, indices in front.items():
        if len(indices) >= 2:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    new_cards = cards[:indices[i]] + cards[indices[i]+1:indices[j]] + cards[indices[j]+1:]
                    if not can_win(new_cards, 1 - turn):
                        return True

    for key, indices in back.items():
        if len(indices) >= 2:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    new_cards = cards[:indices[i]] + cards[indices[i]+1:indices[j]] + cards[indices[j]+1:]
                    if not can_win(new_cards, 1 - turn):
                        return True

    return False

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    cards = []
    for i in range(1, N + 1):
        A_i = int(data[2 * i - 1])
        B_i = int(data[2 * i])
        cards.append((A_i, B_i))

    if can_win(cards, 0):
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == "__main__":
    main()