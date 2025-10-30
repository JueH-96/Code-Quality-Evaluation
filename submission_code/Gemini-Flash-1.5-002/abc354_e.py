# YOUR CODE HERE
def solve():
    n = int(input())
    cards = []
    for _ in range(n):
        cards.append(list(map(int, input().split())))

    def can_win(current_cards):
        if not current_cards:
            return False

        for i in range(len(current_cards)):
            for j in range(i + 1, len(current_cards)):
                if current_cards[i][0] == current_cards[j][0] or current_cards[i][1] == current_cards[j][1]:
                    next_cards = current_cards[:i] + current_cards[i+1:j] + current_cards[j+1:]
                    if not can_win(next_cards):
                        return True
        return False

    if can_win(cards):
        print("Takahashi")
    else:
        print("Aoki")

solve()