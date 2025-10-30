def solve():
    n = int(input())
    cards = []
    for _ in range(n):
        cards.append(list(map(int, input().split())))

    def can_move(current_cards):
        for i in range(len(current_cards)):
            for j in range(i + 1, len(current_cards)):
                if current_cards[i][0] == current_cards[j][0] or \
                   current_cards[i][1] == current_cards[j][1] or \
                   current_cards[i][0] == current_cards[j][1] or \
                   current_cards[i][1] == current_cards[j][0]:
                    return True
        return False

    def winner(current_cards, turn):
        if not can_move(current_cards):
            return "Aoki" if turn == "Takahashi" else "Takahashi"

        if turn == "Takahashi":
            for i in range(len(current_cards)):
                for j in range(i + 1, len(current_cards)):
                    if current_cards[i][0] == current_cards[j][0] or \
                       current_cards[i][1] == current_cards[j][1] or \
                       current_cards[i][0] == current_cards[j][1] or \
                       current_cards[i][1] == current_cards[j][0]:
                        new_cards = current_cards[:i] + current_cards[i+1:j] + current_cards[j+1:]
                        if winner(new_cards, "Aoki") == "Takahashi":
                            return "Takahashi"
            return "Aoki"
        else:
            for i in range(len(current_cards)):
                for j in range(i + 1, len(current_cards)):
                    if current_cards[i][0] == current_cards[j][0] or \
                       current_cards[i][1] == current_cards[j][1] or \
                       current_cards[i][0] == current_cards[j][1] or \
                       current_cards[i][1] == current_cards[j][0]:
                        new_cards = current_cards[:i] + current_cards[i+1:j] + current_cards[j+1:]
                        if winner(new_cards, "Takahashi") == "Aoki":
                            return "Aoki"
            return "Takahashi"

    print(winner(cards, "Takahashi"))

solve()