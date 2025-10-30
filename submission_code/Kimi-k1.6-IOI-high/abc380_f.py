import sys
from functools import lru_cache

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    L = int(input[idx])
    idx += 1
    
    A = list(map(int, input[idx:idx+N]))
    idx += N
    B = list(map(int, input[idx:idx+M]))
    idx += M
    C = list(map(int, input[idx:idx+L]))
    idx += L
    
    A.sort()
    B.sort()
    C.sort()
    
    @lru_cache(maxsize=None)
    def can_win(takahashi, aoki, table, is_takahashi_turn):
        current_hand = takahashi if is_takahashi_turn else aoki
        if not current_hand:
            return False
        
        for card in current_hand:
            # Case 1: Play card without taking any Y
            new_table_list = list(table) + [card]
            new_table_list.sort()
            new_table = tuple(new_table_list)
            
            current_hand_list = list(current_hand)
            current_hand_list.remove(card)
            new_current_hand = tuple(current_hand_list)
            
            if is_takahashi_turn:
                new_takahashi_hand = new_current_hand
                new_aoki_hand = aoki
            else:
                new_aoki_hand = new_current_hand
                new_takahashi_hand = takahashi
            
            if not can_win(new_takahashi_hand, new_aoki_hand, new_table, not is_takahashi_turn):
                return True
            
            # Case 2: Play card and take any Y from old_table that is less than card
            old_table = table
            for y in old_table:
                if y < card:
                    current_hand_list_case2 = list(current_hand)
                    current_hand_list_case2.remove(card)
                    current_hand_list_case2.append(y)
                    current_hand_list_case2.sort()
                    new_current_hand_case2 = tuple(current_hand_list_case2)
                    
                    new_table_list_case2 = list(table)
                    new_table_list_case2.remove(y)
                    new_table_list_case2.append(card)
                    new_table_list_case2.sort()
                    new_table_case2 = tuple(new_table_list_case2)
                    
                    if is_takahashi_turn:
                        new_takahashi_hand_case2 = new_current_hand_case2
                        new_aoki_hand_case2 = aoki
                    else:
                        new_aoki_hand_case2 = new_current_hand_case2
                        new_takahashi_hand_case2 = takahashi
                    
                    if not can_win(new_takahashi_hand_case2, new_aoki_hand_case2, new_table_case2, not is_takahashi_turn):
                        return True
        
        return False
    
    takahashi_tuple = tuple(A)
    aoki_tuple = tuple(B)
    table_tuple = tuple(C)
    result = can_win(takahashi_tuple, aoki_tuple, table_tuple, True)
    print("Takahashi" if result else "Aoki")

if __name__ == "__main__":
    main()