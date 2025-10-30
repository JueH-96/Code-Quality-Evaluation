def who_wins(takahashi_turn, takahashi_hand, aoki_hand, table_cards, memo):
    # Check if the current player has no cards to play
    if takahashi_turn and not takahashi_hand:
        return "Aoki"
    if not takahashi_turn and not aoki_hand:
        return "Takahashi"

    # Create a unique key for the current game state
    key = (takahashi_turn, tuple(sorted(takahashi_hand)), tuple(sorted(aoki_hand)), tuple(sorted(table_cards)))
    
    # Check if we've already computed the result for this state
    if key in memo:
        return memo[key]

    # Current player's hand
    hand = takahashi_hand if takahashi_turn else aoki_hand
    # The result if the current player wins
    win_result = "Takahashi" if takahashi_turn else "Aoki"
    # The result if the current player loses
    lose_result = "Aoki" if takahashi_turn else "Takahashi"

    # Try each card in the current player's hand
    seen_cards = set()
    for card in hand:
        if card in seen_cards:
            continue
        seen_cards.add(card)
        
        # Play the card onto the table
        new_hand = list(hand)
        new_hand.remove(card)
        new_table = list(table_cards) + [card]
        
        # Try not taking any card
        if takahashi_turn:
            next_result = who_wins(not takahashi_turn, new_hand, aoki_hand, new_table, memo)
        else:
            next_result = who_wins(not takahashi_turn, takahashi_hand, new_hand, new_table, memo)
        
        if next_result == win_result:
            memo[key] = win_result
            return win_result
        
        # Try taking a card from the table
        for table_card in table_cards:
            if table_card < card:
                possible_new_hand = list(new_hand) + [table_card]
                possible_new_table = list(new_table)
                possible_new_table.remove(table_card)
                
                if takahashi_turn:
                    next_result = who_wins(not takahashi_turn, possible_new_hand, aoki_hand, possible_new_table, memo)
                else:
                    next_result = who_wins(not takahashi_turn, takahashi_hand, possible_new_hand, possible_new_table, memo)
                
                if next_result == win_result:
                    memo[key] = win_result
                    return win_result

    # If none of the moves lead to a win for the current player, they lose
    memo[key] = lose_result
    return lose_result

# Read input
N, M, L = map(int, input().split())
takahashi_hand = list(map(int, input().split()))
aoki_hand = list(map(int, input().split()))
table_cards = list(map(int, input().split()))

# Determine the winner
result = who_wins(True, takahashi_hand, aoki_hand, table_cards, {})
print(result)