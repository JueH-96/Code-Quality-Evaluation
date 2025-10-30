import sys
import functools
import bisect

# Global constants for turns
TAKAHASHI = 0
AOKI = 1

# Helper to insert a value into a sorted tuple, returning a new sorted tuple.
def insert_into_sorted_tuple(t, val):
    l = list(t)
    bisect.insort_left(l, val) # Inserts val, keeping l sorted.
    return tuple(l)

# Helper to remove a value from a sorted tuple, returning a new sorted tuple.
# Assumes val is in t. If t has multiple occurrences of val, removes the first one.
def remove_from_sorted_tuple(t, val):
    l = list(t)
    l.remove(val) # list.remove(val) removes the first occurrence of val.
    return tuple(l)

@functools.lru_cache(maxsize=None)
def solve(t_hand, a_hand, table, turn):
    # Determine current player's hand, opponent's hand, and next turn
    if turn == TAKAHASHI:
        current_player_original_hand = t_hand
        opponent_original_hand = a_hand
        next_turn = AOKI
        # Base case: Takahashi has no cards to play, so Takahashi loses.
        if not t_hand:
            return False 
    else: # turn == AOKI
        current_player_original_hand = a_hand
        opponent_original_hand = t_hand
        next_turn = TAKAHASHI
        # Base case: Aoki has no cards to play, so Aoki loses.
        if not a_hand:
            return False

    # Iterate over each card the current player can play
    for i in range(len(current_player_original_hand)):
        card_played = current_player_original_hand[i]
        
        # Hand after playing the card (before potentially taking one)
        hand_after_play = current_player_original_hand[:i] + current_player_original_hand[i+1:]
        # Table after the card_played is put on it
        table_after_play = insert_into_sorted_tuple(table, card_played)

        # Option 1: Player plays a card and does NOT take one from the table.
        # The current player wins if the opponent CANNOT win from the resulting state.
        if turn == TAKAHASHI:
            # Takahashi is current player. Next state for Aoki.
            # If solve for Aoki is False (Aoki loses), then Takahashi wins.
            if not solve(hand_after_play, opponent_original_hand, table_after_play, next_turn):
                return True 
        else: # turn == AOKI
            # Aoki is current player. Next state for Takahashi.
            # If solve for Takahashi is False (Takahashi loses), then Aoki wins.
            if not solve(opponent_original_hand, hand_after_play, table_after_play, next_turn):
                return True
        
        # Option 2: Player plays a card AND takes one from the table.
        # The card taken must be strictly smaller than the card played.
        taken_card_values_tried = set() # Optimization: avoid trying to take same value card multiple times if duplicates exist
        for card_idx_on_table in range(len(table_after_play)):
            card_to_take = table_after_play[card_idx_on_table]

            if card_to_take < card_played:
                if card_to_take in taken_card_values_tried:
                    continue 
                taken_card_values_tried.add(card_to_take)

                final_hand = insert_into_sorted_tuple(hand_after_play, card_to_take)
                final_table = remove_from_sorted_tuple(table_after_play, card_to_take)
                
                if turn == TAKAHASHI:
                    if not solve(final_hand, opponent_original_hand, final_table, next_turn):
                        return True 
                else: # turn == AOKI
                    if not solve(opponent_original_hand, final_hand, final_table, next_turn):
                        return True
            else:
                # Since table_after_play is sorted, no further cards will be smaller than card_played.
                break 
    
    # If no move leads to a win for the current player, they lose from this state.
    return False

def main():
    # Read N, M, L (card counts for Takahashi, Aoki, Table)
    # N, M, L are guaranteed to be >= 1.
    N, M, L_val = map(int, sys.stdin.readline().split())
    
    A_list = list(map(int, sys.stdin.readline().split()))
    A_list.sort()
    A = tuple(A_list)

    B_list = list(map(int, sys.stdin.readline().split()))
    B_list.sort()
    B = tuple(B_list)

    C_list = list(map(int, sys.stdin.readline().split()))
    C_list.sort()
    C = tuple(C_list)
        
    if solve(A, B, C, TAKAHASHI):
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == '__main__':
    main()