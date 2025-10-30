import sys
from functools import lru_cache

def main():
    n, m, l = map(int, sys.stdin.readline().split())
    A = sorted(list(map(int, sys.stdin.readline().split())))
    B = sorted(list(map(int, sys.stdin.readline().split())))
    C = sorted(list(map(int, sys.stdin.readline().split())))

    @lru_cache(maxsize=None)
    def dfs(t, a, table, is_t_turn):
        t = list(t)
        a = list(a)
        table = list(table)
        
        if is_t_turn and not t:
            return False
        if not is_t_turn and not a:
            return False
        
        current_hand = t if is_t_turn else a
        for card in current_hand:
            new_current = current_hand.copy()
            new_current.remove(card)
            new_table = table.copy()
            new_table.append(card)
            new_table_sorted = sorted(new_table)
            possible_takes = [c for c in new_table_sorted if c < card]
            
            # Option 1: Take none
            if is_t_turn:
                new_t = tuple(sorted(new_current))
                new_a = tuple(a)
            else:
                new_a = tuple(sorted(new_current))
                new_t = tuple(t)
            next_table = tuple(new_table_sorted)
            if not dfs(new_t, new_a, next_table, not is_t_turn):
                return True
            
            # Option 2: Take each possible_take
            for y in possible_takes:
                new_table_after_take = new_table_sorted.copy()
                new_table_after_take.remove(y)
                new_current_after_take = new_current.copy()
                new_current_after_take.append(y)
                new_current_sorted = sorted(new_current_after_take)
                next_table_take = tuple(sorted(new_table_after_take))
                
                if is_t_turn:
                    new_t_take = tuple(new_current_sorted)
                    new_a_take = tuple(a)
                else:
                    new_a_take = tuple(new_current_sorted)
                    new_t_take = tuple(t)
                if not dfs(new_t_take, new_a_take, next_table_take, not is_t_turn):
                    return True
        
        return False
    
    result = dfs(tuple(A), tuple(B), tuple(C), True)
    print("Takahashi" if result else "Aoki")

if __name__ == "__main__":
    main()