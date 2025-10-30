def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    L = int(data[2])
    
    A = list(map(int, data[3:3+N]))
    B = list(map(int, data[3+N:3+N+M]))
    C = list(map(int, data[3+N+M:3+N+M+L]))
    
    from functools import lru_cache
    
    @lru_cache(None)
    def can_win(takahashi_hand, aoki_hand, table_cards, is_takahashi_turn):
        takahashi_hand = list(takahashi_hand)
        aoki_hand = list(aoki_hand)
        table_cards = list(table_cards)
        
        if is_takahashi_turn:
            if not takahashi_hand:
                return False
            for i in range(len(takahashi_hand)):
                card = takahashi_hand[i]
                new_hand = takahashi_hand[:i] + takahashi_hand[i+1:]
                new_table = table_cards + [card]
                for j in range(len(new_table)):
                    if new_table[j] < card:
                        new_hand.append(new_table[j])
                        new_table = new_table[:j] + new_table[j+1:]
                        break
                if not can_win(tuple(new_hand), tuple(aoki_hand), tuple(new_table), False):
                    return True
            return False
        else:
            if not aoki_hand:
                return True
            for i in range(len(aoki_hand)):
                card = aoki_hand[i]
                new_hand = aoki_hand[:i] + aoki_hand[i+1:]
                new_table = table_cards + [card]
                for j in range(len(new_table)):
                    if new_table[j] < card:
                        new_hand.append(new_table[j])
                        new_table = new_table[:j] + new_table[j+1:]
                        break
                if not can_win(tuple(takahashi_hand), tuple(new_hand), tuple(new_table), True):
                    return True
            return False
    
    if can_win(tuple(A), tuple(B), tuple(C), True):
        print("Takahashi")
    else:
        print("Aoki")