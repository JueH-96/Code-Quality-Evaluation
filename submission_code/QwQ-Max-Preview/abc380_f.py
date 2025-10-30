def main():
    import sys
    from sys import stdin

    sys.setrecursionlimit(1 << 25)
    N, M, L = map(int, stdin.readline().split())
    A = list(map(int, stdin.readline().split())) if N > 0 else []
    B = list(map(int, stdin.readline().split())) if M > 0 else []
    C = list(map(int, stdin.readline().split())) if L > 0 else []
    
    A_tuple = tuple(sorted(A))
    B_tuple = tuple(sorted(B))
    C_tuple = tuple(sorted(C))
    
    initial_state = (A_tuple, B_tuple, C_tuple, 0)
    memo = {}
    
    def can_win(state):
        if state in memo:
            return memo[state]
        
        current_player = state[3]
        if current_player == 0:
            hand = state[0]
            opponent_hand = state[1]
        else:
            hand = state[1]
            opponent_hand = state[0]
        table = state[2]
        
        if not hand:
            memo[state] = False
            return False
        
        for c_played in hand:
            new_hand = list(hand)
            if c_played not in new_hand:
                continue
            new_hand.remove(c_played)
            new_hand_tuple = tuple(new_hand)
            new_table = tuple(sorted(table + (c_played,)))
            eligible = [x for x in new_table if x < c_played]
            
            if not eligible:
                if current_player == 0:
                    next_a = new_hand_tuple
                    next_b = state[1]
                else:
                    next_a = state[0]
                    next_b = new_hand_tuple
                next_state = (next_a, next_b, new_table, 1 - current_player)
                if not can_win(next_state):
                    memo[state] = True
                    return True
            else:
                if current_player == 0:
                    next_a = new_hand_tuple
                    next_b = state[1]
                else:
                    next_a = state[0]
                    next_b = new_hand_tuple
                next_state_take_none = (next_a, next_b, new_table, 1 - current_player)
                if not can_win(next_state_take_none):
                    memo[state] = True
                    return True
                
                for c_taken in eligible:
                    new_table_list = list(new_table)
                    new_table_list.remove(c_taken)
                    new_table_taken = tuple(new_table_list)
                    
                    if current_player == 0:
                        updated_hand = tuple(sorted(new_hand_tuple + (c_taken,)))
                        next_a_hand = updated_hand
                        next_b_hand = state[1]
                    else:
                        updated_hand = tuple(sorted(new_hand_tuple + (c_taken,)))
                        next_a_hand = state[0]
                        next_b_hand = updated_hand
                    next_state = (next_a_hand, next_b_hand, new_table_taken, 1 - current_player)
                    if not can_win(next_state):
                        memo[state] = True
                        return True
        
        memo[state] = False
        return False
    
    result = can_win(initial_state)
    print("Takahashi" if result else "Aoki")

if __name__ == "__main__":
    main()