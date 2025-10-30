from collections import defaultdict
import sys

def can_win(cards, turn):
    if not cards:
        return turn == 1  # Aoki's turn, so Takahashi wins

    # Try removing pairs of cards
    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            if cards[i][0] == cards[j][0] or cards[i][1] == cards[j][1]:
                new_cards = cards[:i] + cards[i+1:j] + cards[j+1:]
                if not can_win(new_cards, 1 - turn):
                    return True
    return False

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    cards = [(int(data[2*i + 1]), int(data[2*i + 2])) for i in range(N)]
    
    if can_win(cards, 0):
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == "__main__":
    main()