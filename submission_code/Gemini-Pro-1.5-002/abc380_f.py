def solve():
    n, m, l = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    c = list(map(int, input().split()))

    memo = {}

    def get_winner(takahashi_hand, aoki_hand, table):
        state = (tuple(sorted(takahashi_hand)), tuple(sorted(aoki_hand)), tuple(sorted(table)))
        if state in memo:
            return memo[state]

        if not takahashi_hand:
            return False
        
        can_win = False
        for card_idx in range(len(takahashi_hand)):
            card = takahashi_hand[card_idx]
            new_table = sorted(table + [card])
            
            for take_idx in range(len(new_table)):
                if new_table[take_idx] < card:
                    new_takahashi_hand = sorted(takahashi_hand[:card_idx] + takahashi_hand[card_idx+1:] + [new_table[take_idx]])
                    new_table_after_take = sorted(new_table[:take_idx] + new_table[take_idx+1:])
                    
                    if not get_winner(aoki_hand, new_takahashi_hand, new_table_after_take):
                        can_win = True
                        break
            else:
                if not get_winner(aoki_hand, takahashi_hand[:card_idx] + takahashi_hand[card_idx+1:], new_table):
                    can_win = True
                    
            if can_win:
                break
        
        memo[state] = can_win
        return can_win

    if get_winner(a, b, c):
        print("Takahashi")
    else:
        print("Aoki")

solve()