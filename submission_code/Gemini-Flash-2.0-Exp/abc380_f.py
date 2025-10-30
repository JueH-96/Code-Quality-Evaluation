def solve():
    n, m, l = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    c = list(map(int, input().split()))

    def can_win(takahashi_hand, aoki_hand, table):
        if not takahashi_hand:
            return False

        for i in range(len(takahashi_hand)):
            new_table = table + [takahashi_hand[i]]
            new_takahashi_hand = takahashi_hand[:i] + takahashi_hand[i+1:]
            
            taken_cards = []
            for j in range(len(new_table) - 1):
                if new_table[j] < new_table[-1]:
                    taken_cards.append(new_table[j])
            
            new_table = [card for card in new_table[:-1] if card >= new_table[-1]] + [new_table[-1]]
            new_takahashi_hand += taken_cards
            
            if not can_win_aoki(new_takahashi_hand, aoki_hand, new_table):
                return True
        return False

    def can_win_aoki(takahashi_hand, aoki_hand, table):
        if not aoki_hand:
            return False

        for i in range(len(aoki_hand)):
            new_table = table + [aoki_hand[i]]
            new_aoki_hand = aoki_hand[:i] + aoki_hand[i+1:]
            
            taken_cards = []
            for j in range(len(new_table) - 1):
                if new_table[j] < new_table[-1]:
                    taken_cards.append(new_table[j])
            
            new_table = [card for card in new_table[:-1] if card >= new_table[-1]] + [new_table[-1]]
            new_aoki_hand += taken_cards

            if not can_win(takahashi_hand, new_aoki_hand, new_table):
                return True
        return False

    if can_win(a, b, c):
        print("Takahashi")
    else:
        print("Aoki")

solve()