def determine_winner():
    import sys
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    cards = [(int(data[i]), int(data[i + 1])) for i in range(1, 2 * N, 2)]

    def can_win(cards, turn):
        for i in range(len(cards)):
            for j in range(i + 1, len(cards)):
                if cards[i][0] == cards[j][0] or cards[i][1] == cards[j][1]:
                    new_cards = cards[:i] + cards[i+1:j] + cards[j+1:]
                    if not can_win(new_cards, not turn):
                        return True
        return False

    if can_win(cards, True):
        print("Takahashi")
    else:
        print("Aoki")

determine_winner()