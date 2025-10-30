import sys
from functools import lru_cache

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    L = int(input[ptr])
    ptr += 1
    
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+M]))
    ptr += M
    C = list(map(int, input[ptr:ptr+L]))
    ptr += L
    
    A.sort()
    B.sort()
    C.sort()
    
    A = tuple(A)
    B = tuple(B)
    C = tuple(C)
    
    @lru_cache(maxsize=None)
    def can_win(turn, a_hand, b_hand, table):
        current_hand = a_hand if turn == 0 else b_hand
        if not current_hand:
            return False
        
        for x in current_hand:
            new_table_list = list(table) + [x]
            new_table_list.sort()
            new_table = tuple(new_table_list)
            
            current_hand_list = list(current_hand)
            current_hand_list.remove(x)
            new_hand = tuple(current_hand_list)
            
            if turn == 0:
                new_a = new_hand
                new_b = b_hand
            else:
                new_a = a_hand
                new_b = new_hand
            
            if not can_win(1 - turn, new_a, new_b, new_table):
                return True
            
            candidates = [y for y in new_table if y < x]
            for y in candidates:
                new_table_after_take = list(new_table)
                try:
                    new_table_after_take.remove(y)
                except ValueError:
                    pass
                new_table_after_take = tuple(new_table_after_take)
                
                current_hand_list = list(current_hand)
                current_hand_list.remove(x)
                current_hand_list.append(y)
                current_hand_list.sort()
                new_hand_with_y = tuple(current_hand_list)
                
                if turn == 0:
                    new_a = new_hand_with_y
                    new_b = b_hand
                else:
                    new_a = a_hand
                    new_b = new_hand_with_y
                
                if not can_win(1 - turn, new_a, new_b, new_table_after_take):
                    return True
        
        return False
    
    result = can_win(0, A, B, C)
    print("Takahashi" if result else "Aoki")

if __name__ == "__main__":
    main()