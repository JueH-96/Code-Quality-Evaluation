def solve():
    n = int(input())
    initial_cards = []
    for _ in range(n):
        a, b = map(int, input().split())
        initial_cards.append((a, b))
    
    memo = {}
    
    def get_possible_moves(current_cards):
        possible_next_states = []
        num_cards = len(current_cards)
        if num_cards < 2:
            return []
        for i in range(num_cards):
            for j in range(i + 1, num_cards):
                card1 = current_cards[i]
                card2 = current_cards[j]
                if card1[0] == card2[0] or card1[1] == card2[1]:
                    next_cards_list = []
                    for k in range(num_cards):
                        if k != i and k != j:
                            next_cards_list.append(current_cards[k])
                    possible_next_states.append(tuple(sorted(next_cards_list)))
        return set(possible_next_states)

    def can_win(current_cards_tuple):
        if current_cards_tuple in memo:
            return memo[current_cards_tuple]
        
        next_states = get_possible_moves(list(current_cards_tuple))
        if not next_states:
            result = False
        else:
            winning_move_found = False
            for next_state_tuple in next_states:
                if not can_win(next_state_tuple):
                    winning_move_found = True
                    break
            result = winning_move_found
            
        memo[current_cards_tuple] = result
        return result

    initial_state_tuple = tuple(sorted(initial_cards))
    if can_win(initial_state_tuple):
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == '__main__':
    solve()