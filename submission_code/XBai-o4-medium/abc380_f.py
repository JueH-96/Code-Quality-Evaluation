import sys
from functools import lru_cache

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr += 1
    M = int(input[ptr]); ptr += 1
    L = int(input[ptr]); ptr += 1
    
    A = list(map(int, input[ptr:ptr+N])); ptr += N
    B = list(map(int, input[ptr:ptr+M])); ptr += M
    C = list(map(int, input[ptr:ptr+L])); ptr += L
    
    t_hand = tuple(sorted(A))
    a_hand = tuple(sorted(B))
    table = tuple(sorted(C))
    
    @lru_cache(maxsize=None)
    def dfs(t_hand, a_hand, table, is_t_turn):
        current_hand = t_hand if is_t_turn else a_hand
        if not current_hand:
            return False
        
        unique_values = []
        prev = None
        for val in current_hand:
            if val != prev:
                unique_values.append(val)
                prev = val
        
        for play_val in unique_values:
            current_hand_list = list(current_hand)
            idx = current_hand_list.index(play_val)
            new_current_hand_list = current_hand_list[:idx] + current_hand_list[idx+1:]
            new_current_hand = tuple(new_current_hand_list)
            
            new_table_list = list(table) + [play_val]
            new_table_list.sort()
            new_table = tuple(new_table_list)
            
            original_table = table
            
            possible_takes = []
            prev_take = None
            for c in original_table:
                if c < play_val:
                    if c != prev_take:
                        possible_takes.append(c)
                        prev_take = c
            
            if not possible_takes:
                if is_t_turn:
                    next_t = new_current_hand
                    next_a = a_hand
                else:
                    next_t = t_hand
                    next_a = new_current_hand
                next_turn = not is_t_turn
                if not dfs(next_t, next_a, new_table, next_turn):
                    return True
            else:
                for take_val in possible_takes:
                    original_table_list = list(original_table)
                    take_idx = original_table_list.index(take_val)
                    new_original_table_list = original_table_list[:take_idx] + original_table_list[take_idx+1:]
                    
                    new_table_after_take = new_original_table_list + [play_val]
                    new_table_after_take.sort()
                    new_table_after_take_tuple = tuple(new_table_after_take)
                    
                    new_current_hand_after_take_list = list(new_current_hand) + [take_val]
                    new_current_hand_after_take_list.sort()
                    new_current_hand_after_take = tuple(new_current_hand_after_take_list)
                    
                    if is_t_turn:
                        next_t = new_current_hand_after_take
                        next_a = a_hand
                    else:
                        next_t = t_hand
                        next_a = new_current_hand_after_take
                    
                    next_turn = not is_t_turn
                    
                    if not dfs(next_t, next_a, new_table_after_take_tuple, next_turn):
                        return True
        return False
    
    result = dfs(t_hand, a_hand, table, True)
    print("Takahashi" if result else "Aoki")

if __name__ == "__main__":
    main()