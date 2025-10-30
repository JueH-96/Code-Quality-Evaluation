def solve_game():
    n, m, l = map(int, input().split())
    a_cards = list(map(int, input().split()))
    b_cards = list(map(int, input().split()))
    c_cards = list(map(int, input().split()))
    
    memo = {}
    
    def get_state_tuple(is_takahashi_turn, takahashi_hand, aoki_hand, table_cards):
        return (is_takahashi_turn, tuple(sorted(takahashi_hand)), tuple(sorted(aoki_hand)), tuple(sorted(table_cards)))
        
    def is_winning(is_takahashi_turn, takahashi_hand, aoki_hand, table_cards):
        state = get_state_tuple(is_takahashi_turn, takahashi_hand, aoki_hand, table_cards)
        if state in memo:
            return memo[state]
        
        current_hand = takahashi_hand if is_takahashi_turn else aoki_hand
        if not current_hand:
            return False
            
        possible_moves_exist = False
        
        for card_to_play in list(current_hand):
            possible_moves_exist = True
            remaining_hand = list(current_hand)
            remaining_hand.remove(card_to_play)
            next_table_cards = list(table_cards)
            next_table_cards.append(card_to_play)
            cards_less_than_played = [c for c in table_cards if c < card_to_play]
            
            next_states_are_losing_for_opponent = False
            
            # Option 1: Don't take any card from table
            next_takahashi_hand_no_take = remaining_hand if is_takahashi_turn else takahashi_hand
            next_aoki_hand_no_take = remaining_hand if not is_takahashi_turn else aoki_hand
            if not is_winning(not is_takahashi_turn, next_takahashi_hand_no_take, next_aoki_hand_no_take, tuple(sorted(next_table_cards))):
                next_states_are_losing_for_opponent = True
                
            if cards_less_than_played:
                for card_to_take in cards_less_than_played:
                    next_takahashi_hand_take = list(remaining_hand) if is_takahashi_turn else list(takahashi_hand)
                    next_aoki_hand_take = list(remaining_hand) if not is_takahashi_turn else list(aoki_hand)
                    next_takahashi_hand_take.append(card_to_take)
                    next_table_cards_take = list(next_table_cards)
                    next_table_cards_take.remove(card_to_take)
                    if not is_winning(not is_takahashi_turn, next_takahashi_hand_take, next_aoki_hand_take, tuple(sorted(next_table_cards_take))):
                        next_states_are_losing_for_opponent = True
                        
            if next_states_are_losing_for_opponent:
                memo[state] = True
                return True
                
        memo[state] = False
        return False
        
    initial_takahashi_hand = tuple(sorted(a_cards))
    initial_aoki_hand = tuple(sorted(b_cards))
    initial_table_cards = tuple(sorted(c_cards))
    
    if is_winning(True, list(initial_takahashi_hand), list(initial_aoki_hand), list(initial_table_cards)):
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == '__main__':
    solve_game()