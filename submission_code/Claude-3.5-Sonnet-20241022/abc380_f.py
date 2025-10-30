def play_game(n, m, l, a, b, c, turn, memo):
    state = (tuple(sorted(a)), tuple(sorted(b)), tuple(sorted(c)), turn)
    if state in memo:
        return memo[state]
    
    # Check if current player has no cards
    if turn == 0 and not a:
        return False
    if turn == 1 and not b:
        return False
        
    current_hand = a if turn == 0 else b
    result = False
    
    # Try each card from current player's hand
    for i, card in enumerate(current_hand):
        new_hand = list(current_hand)
        new_hand.pop(i)
        new_table = list(c) + [card]
        
        # Try taking a card from table
        can_take = [j for j, table_card in enumerate(c) if table_card < card]
        
        # Don't take any card
        if not play_game(
            list(new_hand) if turn == 0 else a,
            list(new_hand) if turn == 1 else b,
            new_table,
            1 - turn,
            memo
        ):
            result = True
            break
            
        # Try taking each possible card
        for j in can_take:
            take_table = list(new_table)
            take_card = take_table.pop(j)
            new_hand.append(take_card)
            
            if not play_game(
                list(new_hand) if turn == 0 else a,
                list(new_hand) if turn == 1 else b,
                take_table,
                1 - turn,
                memo
            ):
                result = True
                break
                
            new_hand.pop()
            
        if result:
            break
            
    memo[state] = result
    return result

def solve():
    # Read input
    n, m, l = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    c = list(map(int, input().split()))
    
    # Play game with memoization
    memo = {}
    takahashi_wins = play_game(n, m, l, a, b, c, 0, memo)
    
    print("Takahashi" if takahashi_wins else "Aoki")

solve()