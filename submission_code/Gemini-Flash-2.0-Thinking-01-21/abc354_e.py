def solve():
    n = int(input())
    cards = []
    for _ in range(n):
        a, b = map(int, input().split())
        cards.append({'front': a, 'back': b})
    
    memo = {}
    
    def get_winning_state(current_card_indices):
        state_tuple = tuple(sorted(current_card_indices))
        if state_tuple in memo:
            return memo[state_tuple]
        
        possible_moves = []
        indices_list = list(current_card_indices)
        for i_index in range(len(indices_list)):
            for j_index in range(i_index + 1, len(indices_list)):
                card_index1 = indices_list[i_index]
                card_index2 = indices_list[j_index]
                if cards[card_index1-1]['front'] == cards[card_index2-1]['front'] or cards[card_index1-1]['back'] == cards[card_index2-1]['back']:
                    possible_moves.append((card_index1, card_index2))
                    
        if not possible_moves:
            result = False
        else:
            win_found = False
            for move in possible_moves:
                card1_index, card2_index = move
                next_card_indices = set(current_card_indices)
                next_card_indices.remove(card1_index)
                next_card_indices.remove(card2_index)
                if not get_winning_state(next_card_indices):
                    win_found = True
                    break
            result = win_found
            
        memo[state_tuple] = result
        return result
        
    initial_card_indices = set(range(1, n + 1))
    if get_winning_state(initial_card_indices):
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == '__main__':
    solve()