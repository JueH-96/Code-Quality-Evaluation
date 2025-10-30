def solve_game():
    n, m, l = map(int, input().split())
    a_hand = list(map(int, input().split()))
    b_hand = list(map(int, input().split()))
    table_cards = list(map(int, input().split()))
    
    memo = {}
    
    def get_game_result(current_takahashi_hand, current_aoki_hand, current_table_cards, is_takahashi_turn):
        takahashi_hand_tuple = tuple(sorted(current_takahashi_hand))
        aoki_hand_tuple = tuple(sorted(current_aoki_hand))
        table_cards_tuple = tuple(sorted(current_table_cards))
        state = (takahashi_hand_tuple, aoki_hand_tuple, table_cards_tuple, is_takahashi_turn)
        if state in memo:
            return memo[state]
        
        if is_takahashi_turn:
            if not current_takahashi_hand:
                result = False # Takahashi cannot move, loses
                memo[state] = result
                return result
            
            possible_win = False
            for card_index in range(len(current_takahashi_hand)):
                card_to_play = current_takahashi_hand[card_index]
                next_takahashi_hand = current_takahashi_hand[:card_index] + current_takahashi_hand[card_index+1:]
                next_table_cards = current_table_cards + [card_to_play]
                takeable_cards = [c for c in next_table_cards if c < card_to_play]
                
                # Option 1: Don't take any card
                if not get_game_result(next_takahashi_hand, current_aoki_hand, next_table_cards, False):
                    possible_win = True
                    break
                    
                # Option 2: Take a card if possible
                for card_to_take in takeable_cards:
                    temp_table_cards = list(next_table_cards)
                    temp_table_cards.remove(card_to_take)
                    next_takahashi_hand_with_taken_card = next_takahashi_hand + [card_to_take]
                    if not get_game_result(next_takahashi_hand_with_taken_card, current_aoki_hand, temp_table_cards, False):
                        possible_win = True
                        break
                if possible_win:
                    break
                    
            result = possible_win
            memo[state] = result
            return result
        else: # Aoki's turn
            if not current_aoki_hand:
                result = False # Aoki cannot move, loses
                memo[state] = result
                return result
                
            possible_win = False
            for card_index in range(len(current_aoki_hand)):
                card_to_play = current_aoki_hand[card_index]
                next_aoki_hand = current_aoki_hand[:card_index] + current_aoki_hand[card_index+1:]
                next_table_cards = current_table_cards + [card_to_play]
                takeable_cards = [c for c in next_table_cards if c < card_to_play]
                
                # Option 1: Don't take any card
                if not get_game_result(current_takahashi_hand, next_aoki_hand, next_table_cards, True):
                    possible_win = True
                    break
                    
                # Option 2: Take a card if possible
                for card_to_take in takeable_cards:
                    temp_table_cards = list(next_table_cards)
                    temp_table_cards.remove(card_to_take)
                    next_aoki_hand_with_taken_card = next_aoki_hand + [card_to_take]
                    if not get_game_result(current_takahashi_hand, next_aoki_hand_with_taken_card, temp_table_cards, True):
                        possible_win = True
                        break
                if possible_win:
                    break
                    
            result = possible_win
            memo[state] = result
            return result
            
    if get_game_result(a_hand, b_hand, table_cards, True):
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == '__main__':
    solve_game()