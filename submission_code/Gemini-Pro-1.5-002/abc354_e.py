# YOUR CODE HERE
def solve():
    n = int(input())
    cards = []
    for _ in range(n):
        cards.append(list(map(int, input().split())))

    def can_move(current_cards):
        for i in range(len(current_cards)):
            for j in range(i + 1, len(current_cards)):
                if current_cards[i][0] == current_cards[j][0] or current_cards[i][1] == current_cards[j][1]:
                    return True
        return False

    def winner(current_cards, turn):
        if not can_move(current_cards):
            return turn == 1
        
        for i in range(len(current_cards)):
            for j in range(i + 1, len(current_cards)):
                if current_cards[i][0] == current_cards[j][0] or current_cards[i][1] == current_cards[j][1]:
                    next_cards = current_cards[:i] + current_cards[i+1:j] + current_cards[j+1:]
                    if not winner(next_cards, 1 - turn):
                        return True
        return False

    if winner(cards, 0):
        print("Takahashi")
    else:
        print("Aoki")

solve()