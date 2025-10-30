from itertools import combinations

def can_win(cards, turn):
    if not cards:
        return turn
    for a, b in combinations(cards, 2):
        if a[0] == b[0] or a[1] == b[1]:
            new_cards = cards.copy()
            new_cards.remove(a)
            new_cards.remove(b)
            if not can_win(new_cards, not turn):
                return turn
    return not turn

def main():
    N = int(input().strip())
    cards = [tuple(map(int, input().split())) for _ in range(N)]
    winner = can_win(cards, True)
    print("Takahashi" if winner else "Aoki")

if __name__ == "__main__":
    main()