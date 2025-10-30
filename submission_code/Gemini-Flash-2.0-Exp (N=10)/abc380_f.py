def solve():
    n, m, l = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    c = list(map(int, input().split()))

    def winner(takahashi_hand, aoki_hand, table, turn):
        if turn == 0: # Takahashi's turn
            if not takahashi_hand:
                return "Aoki"
            
            best_result = "Aoki"
            for i in range(len(takahashi_hand)):
                new_table = table[:]
                new_table.append(takahashi_hand[i])
                new_takahashi_hand = takahashi_hand[:i] + takahashi_hand[i+1:]
                
                taken_cards = []
                for j in range(len(new_table) - 1):
                    if new_table[j] < new_table[-1]:
                        taken_cards.append(new_table[j])
                
                new_table = [card for card in new_table if card not in taken_cards]
                new_takahashi_hand.extend(taken_cards)
                
                result = winner(new_takahashi_hand, aoki_hand, new_table, 1)
                if result == "Takahashi":
                    best_result = "Takahashi"
                    break
            return best_result
        else: # Aoki's turn
            if not aoki_hand:
                return "Takahashi"
            
            best_result = "Takahashi"
            for i in range(len(aoki_hand)):
                new_table = table[:]
                new_table.append(aoki_hand[i])
                new_aoki_hand = aoki_hand[:i] + aoki_hand[i+1:]
                
                taken_cards = []
                for j in range(len(new_table) - 1):
                    if new_table[j] < new_table[-1]:
                        taken_cards.append(new_table[j])
                
                new_table = [card for card in new_table if card not in taken_cards]
                new_aoki_hand.extend(taken_cards)
                
                result = winner(takahashi_hand, new_aoki_hand, new_table, 0)
                if result == "Aoki":
                    best_result = "Aoki"
                    break
            return best_result

    print(winner(a, b, c, 0))

solve()