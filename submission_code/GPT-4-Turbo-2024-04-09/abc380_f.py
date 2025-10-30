def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    L = int(data[2])
    
    A = list(map(int, data[3:3+N]))
    B = list(map(int, data[3+N:3+N+M]))
    C = list(map(int, data[3+N+M:]))
    
    from functools import lru_cache
    
    # Sort the cards to make it easier to manage
    A.sort()
    B.sort()
    C.sort()
    
    # Convert lists to tuples for hashing in lru_cache
    A = tuple(A)
    B = tuple(B)
    C = tuple(C)
    
    # Memoization of game states
    @lru_cache(None)
    def dp(takahashi, aoki, table):
        # Convert back to lists for manipulation
        takahashi = list(takahashi)
        aoki = list(aoki)
        table = list(table)
        
        # Determine whose turn it is by the length of the game state
        takahashi_turn = (len(takahashi) + len(aoki) + len(table)) % 2 == 0
        
        if takahashi_turn:
            # Takahashi's turn
            if not takahashi:
                return False  # Takahashi cannot play, loses
            
            for i, card in enumerate(takahashi):
                new_takahashi = takahashi[:i] + takahashi[i+1:]
                new_table = table + [card]
                new_table.sort()
                
                # Check if Takahashi can pick a card from the table
                for j, t_card in enumerate(new_table):
                    if t_card < card:
                        # Takahashi picks this card
                        new_takahashi.append(t_card)
                        new_takahashi.sort()
                        new_table = new_table[:j] + new_table[j+1:]
                        break
                
                # Recurse with new state
                if dp(tuple(new_takahashi), tuple(aoki), tuple(new_table)):
                    return True
            return False
        else:
            # Aoki's turn
            if not aoki:
                return True  # Aoki cannot play, Takahashi wins
            
            for i, card in enumerate(aoki):
                new_aoki = aoki[:i] + aoki[i+1:]
                new_table = table + [card]
                new_table.sort()
                
                # Check if Aoki can pick a card from the table
                for j, t_card in enumerate(new_table):
                    if t_card < card:
                        # Aoki picks this card
                        new_aoki.append(t_card)
                        new_aoki.sort()
                        new_table = new_table[:j] + new_table[j+1:]
                        break
                
                # Recurse with new state
                if not dp(tuple(takahashi), tuple(new_aoki), tuple(new_table)):
                    return False
            return True
    
    # Start the game with Takahashi's turn
    if dp(tuple(A), tuple(B), tuple(C)):
        print("Takahashi")
    else:
        print("Aoki")