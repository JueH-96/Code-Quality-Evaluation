import sys
from functools import lru_cache

def main():
    sys.setrecursionlimit(10000)
    N, M, L = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    C = list(map(int, sys.stdin.readline().split()))
    
    A.sort()
    B.sort()
    C.sort()
    
    @lru_cache(maxsize=None)
    def can_win(my_hand, their_hand, table):
        my_hand_list = list(my_hand)
        prev_x = None
        for i in range(len(my_hand_list)):
            x = my_hand_list[i]
            if x == prev_x:
                continue
            prev_x = x
            
            # Remove x from my_hand
            new_my_hand_list = list(my_hand_list)
            new_my_hand_list.remove(x)
            
            # Add x to table and sort
            new_table_after_play = list(table) + [x]
            new_table_after_play.sort()
            
            # Possible Ys (all <x)
            possible_Ys = []
            for y in new_table_after_play:
                if y < x:
                    possible_Ys.append(y)
                else:
                    break
            
            # Consider not taking any Y (even if possible)
            new_state = (their_hand, tuple(new_my_hand_list), tuple(new_table_after_play))
            if not can_win(*new_state):
                return True
            
            # Consider all possible Ys
            prev_y = None
            for y in possible_Ys:
                if y == prev_y:
                    continue
                prev_y = y
                
                # Create new_my_hand_plus_y
                new_my_hand_plus_y_list = list(new_my_hand_list) + [y]
                new_my_hand_plus_y_list.sort()
                new_my_hand_plus_y_tuple = tuple(new_my_hand_plus_y_list)
                
                # Create new_table_after_take
                new_table_after_take = list(new_table_after_play)
                new_table_after_take.remove(y)
                new_table_after_take_tuple = tuple(new_table_after_take)
                
                new_state_y = (their_hand, new_my_hand_plus_y_tuple, new_table_after_take_tuple)
                if not can_win(*new_state_y):
                    return True
        
        # No winning move found
        return False
    
    result = can_win(tuple(A), tuple(B), tuple(C))
    print("Takahashi" if result else "Aoki")

if __name__ == "__main__":
    main()