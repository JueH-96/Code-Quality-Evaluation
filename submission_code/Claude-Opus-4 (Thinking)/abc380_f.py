# YOUR CODE HERE
N, M, L = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
C = list(map(int, input().split()))

# Create a list of all cards
all_cards = A + B + C
n_cards = len(all_cards)

# Initial state: Takahashi has first N cards, Aoki has next M cards, table has last L cards
takahashi_mask = (1 << N) - 1
aoki_mask = ((1 << M) - 1) << N
table_mask = ((1 << L) - 1) << (N + M)

# Memoization
memo = {}

def can_win(takahashi_mask, aoki_mask, table_mask, is_takahashi_turn):
    state = (takahashi_mask, aoki_mask, table_mask, is_takahashi_turn)
    if state in memo:
        return memo[state]
    
    if is_takahashi_turn:
        # Takahashi's turn
        if takahashi_mask == 0:
            # Takahashi has no cards, loses
            memo[state] = False
            return False
        
        # Try all possible moves
        for i in range(n_cards):
            if takahashi_mask & (1 << i):
                # Card i is in Takahashi's hand
                card_value = all_cards[i]
                
                # Put card i on the table
                new_takahashi = takahashi_mask ^ (1 << i)
                new_table = table_mask | (1 << i)
                
                # Option 1: Don't take any card
                if not can_win(new_takahashi, aoki_mask, new_table, False):
                    # If Aoki can't win from this state, Takahashi wins
                    memo[state] = True
                    return True
                
                # Option 2: Take a card if possible
                for j in range(n_cards):
                    if new_table & (1 << j) and j != i and all_cards[j] < card_value:
                        # Can take card j
                        new_takahashi2 = new_takahashi | (1 << j)
                        new_table2 = new_table ^ (1 << j)
                        
                        if not can_win(new_takahashi2, aoki_mask, new_table2, False):
                            # If Aoki can't win from this state, Takahashi wins
                            memo[state] = True
                            return True
        
        # No winning move found
        memo[state] = False
        return False
    else:
        # Aoki's turn
        if aoki_mask == 0:
            # Aoki has no cards, loses
            memo[state] = False
            return False
        
        # Try all possible moves
        for i in range(n_cards):
            if aoki_mask & (1 << i):
                # Card i is in Aoki's hand
                card_value = all_cards[i]
                
                # Put card i on the table
                new_aoki = aoki_mask ^ (1 << i)
                new_table = table_mask | (1 << i)
                
                # Option 1: Don't take any card
                if not can_win(takahashi_mask, new_aoki, new_table, True):
                    # If Takahashi can't win from this state, Aoki wins
                    memo[state] = True
                    return True
                
                # Option 2: Take a card if possible
                for j in range(n_cards):
                    if new_table & (1 << j) and j != i and all_cards[j] < card_value:
                        # Can take card j
                        new_aoki2 = new_aoki | (1 << j)
                        new_table2 = new_table ^ (1 << j)
                        
                        if not can_win(takahashi_mask, new_aoki2, new_table2, True):
                            # If Takahashi can't win from this state, Aoki wins
                            memo[state] = True
                            return True
        
        # No winning move found
        memo[state] = False
        return False

if can_win(takahashi_mask, aoki_mask, table_mask, True):
    print("Takahashi")
else:
    print("Aoki")