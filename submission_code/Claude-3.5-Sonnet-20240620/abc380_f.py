# YOUR CODE HERE
def play_game(hand1, hand2, table):
    if not hand1:
        return "Aoki"
    if not hand2:
        return "Takahashi"
    
    for card in hand1:
        new_hand1 = hand1.copy()
        new_hand1.remove(card)
        new_table = table + [card]
        
        take_card = None
        for t_card in table:
            if t_card < card:
                if take_card is None or t_card > take_card:
                    take_card = t_card
        
        if take_card is not None:
            new_hand1.append(take_card)
            new_table.remove(take_card)
        
        if play_game(hand2, new_hand1, new_table) == "Aoki":
            return "Takahashi"
    
    return "Aoki"

# Read input
N, M, L = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
C = list(map(int, input().split()))

# Solve and print result
result = play_game(A, B, C)
print(result)