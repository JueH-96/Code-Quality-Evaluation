from functools import lru_cache

def solve():
    import sys
    N, M, L = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    C = list(map(int, sys.stdin.readline().split()))
    
    table_initial = frozenset(C)
    tak_hand_initial = frozenset(A)
    aoki_hand_initial = frozenset(B)
    initial_state = (table_initial, tak_hand_initial, aoki_hand_initial, 'T')
    
    @lru_cache(maxsize=None)
    def can_win(state):
        table = state[0]
        tak_hand = state[1]
        aoki_hand = state[2]
        player = state[3]
        
        if not tak_hand and player == 'T':
            return False
        if not aoki_hand and player == 'A':
            return False
        
        if player == 'T':
            current_hand = tak_hand
            opponent_hand = aoki_hand
            opponent_table = table
        else:
            current_hand = aoki_hand
            opponent_hand = tak_hand
            opponent_table = table
        
        for x in current_hand:
            new_table = set(opponent_table)
            new_table.add(x)
            new_table = frozenset(new_table)
            
            new_tak_hand = tak_hand - frozenset([x]) if player == 'T' else aoki_hand - frozenset([x])
            
            new_state_case1 = (new_table, new_tak_hand, aoki_hand if player == 'T' else tak_hand, 'A' if player == 'T' else 'T')
            if not can_win(new_state_case1):
                return True
            
            for y in new_table:
                if y < x:
                    new_table_case2 = new_table - frozenset([y])
                    new_tak_hand_case2 = set(new_tak_hand)
                    new_tak_hand_case2.add(y)
                    new_tak_hand_case2 = frozenset(new_tak_hand_case2)
                    
                    new_state_case2 = (new_table_case2, new_tak_hand_case2, aoki_hand if player == 'T' else tak_hand, 'A' if player == 'T' else 'T')
                    if not can_win(new_state_case2):
                        return True
        
        return False
    
    result = can_win(initial_state)
    if result:
        print("Takahashi")
    else:
        print("Aoki")

solve()