from functools import lru_cache

def solve():
    N, M, L = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    C = list(map(int, input().split()))
    
    # All cards with their values
    all_cards = A + B + C
    
    # Initial state
    takahashi_cards = frozenset(range(N))
    aoki_cards = frozenset(range(N, N + M))
    table_cards = frozenset(range(N + M, N + M + L))
    
    @lru_cache(maxsize=None)
    def can_win(turn, tak_cards, aoki_cards, table_cards):
        # turn: 0 for Takahashi, 1 for Aoki
        current_hand = tak_cards if turn == 0 else aoki_cards
        
        if not current_hand:
            # Current player has no cards, loses
            return False
        
        # Try all possible moves
        for card_idx in current_hand:
            card_value = all_cards[card_idx]
            
            # Play this card to the table
            new_table = table_cards | {card_idx}
            new_tak_cards = tak_cards - {card_idx} if turn == 0 else tak_cards
            new_aoki_cards = aoki_cards - {card_idx} if turn == 1 else aoki_cards
            
            # Generate all possible next states after this move
            possible_next_states = []
            
            # Option 1: Don't take any card from table
            possible_next_states.append((new_tak_cards, new_aoki_cards, new_table))
            
            # Option 2: Take one card from table (if possible)
            for table_card_idx in table_cards:
                table_card_value = all_cards[table_card_idx]
                if table_card_value < card_value:
                    # Current player takes this card
                    newer_table = new_table - {table_card_idx}
                    if turn == 0:  # Takahashi takes the card
                        newer_tak_cards = new_tak_cards | {table_card_idx}
                        newer_aoki_cards = new_aoki_cards
                    else:  # Aoki takes the card
                        newer_tak_cards = new_tak_cards
                        newer_aoki_cards = new_aoki_cards | {table_card_idx}
                    possible_next_states.append((newer_tak_cards, newer_aoki_cards, newer_table))
            
            # Check if any next state leads to opponent losing
            for next_tak, next_aoki, next_table in possible_next_states:
                if not can_win(1 - turn, next_tak, next_aoki, next_table):
                    return True
        
        # If no winning move found
        return False
    
    # Takahashi starts first (turn = 0)
    if can_win(0, takahashi_cards, aoki_cards, table_cards):
        print("Takahashi")
    else:
        print("Aoki")

solve()