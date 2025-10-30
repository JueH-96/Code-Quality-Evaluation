def solve():
    N, M, L = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    C = list(map(int, input().split()))
    
    # Combine all cards and sort them
    all_cards = []
    for i in range(N):
        all_cards.append((A[i], 0, i))  # (value, type, index)
    for i in range(M):
        all_cards.append((B[i], 1, i))
    for i in range(L):
        all_cards.append((C[i], 2, i))
    
    total = N + M + L
    
    # Create initial state masks
    takahashi_mask = 0
    aoki_mask = 0
    table_mask = 0
    
    for i in range(total):
        val, typ, idx = all_cards[i]
        if typ == 0:
            takahashi_mask |= (1 << i)
        elif typ == 1:
            aoki_mask |= (1 << i)
        else:
            table_mask |= (1 << i)
    
    # Memoization
    memo = {}
    
    def can_win(tak_mask, aoki_mask, table_mask, is_takahashi_turn):
        # Check if current player has no cards (loses)
        if is_takahashi_turn and tak_mask == 0:
            return False
        if not is_takahashi_turn and aoki_mask == 0:
            return False
        
        state = (tak_mask, aoki_mask, table_mask, is_takahashi_turn)
        if state in memo:
            return memo[state]
        
        current_mask = tak_mask if is_takahashi_turn else aoki_mask
        
        # Try all possible moves
        for i in range(total):
            if not (current_mask & (1 << i)):
                continue
            
            # Play card i
            new_tak = tak_mask
            new_aoki = aoki_mask
            new_table = table_mask | (1 << i)
            
            if is_takahashi_turn:
                new_tak ^= (1 << i)
            else:
                new_aoki ^= (1 << i)
            
            # Try taking no card from table
            if not can_win(new_tak, new_aoki, new_table, not is_takahashi_turn):
                memo[state] = True
                return True
            
            # Try taking each smaller card from table
            for j in range(total):
                if not (new_table & (1 << j)):
                    continue
                if all_cards[j][0] >= all_cards[i][0]:
                    continue
                
                # Take card j from table
                final_tak = new_tak
                final_aoki = new_aoki
                final_table = new_table ^ (1 << j)
                
                if is_takahashi_turn:
                    final_tak |= (1 << j)
                else:
                    final_aoki |= (1 << j)
                
                if not can_win(final_tak, final_aoki, final_table, not is_takahashi_turn):
                    memo[state] = True
                    return True
        
        memo[state] = False
        return False
    
    # Check if Takahashi can win from initial state
    if can_win(takahashi_mask, aoki_mask, table_mask, True):
        print("Takahashi")
    else:
        print("Aoki")

solve()