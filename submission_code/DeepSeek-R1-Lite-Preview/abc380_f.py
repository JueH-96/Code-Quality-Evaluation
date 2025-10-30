from functools import lru_cache

def main():
    import sys
    sys.setrecursionlimit(1000000)
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    L = int(data[2])
    
    index = 3
    A = [int(data[index + i]) for i in range(N)]
    index += N
    B = [int(data[index + i]) for i in range(M)]
    index += M
    C = [int(data[index + i]) for i in range(L)]
    
    # Assign unique IDs to each card
    value_map = {}
    current_id = 1
    
    # Takahashi's cards
    t_ids = set()
    for value in A:
        t_ids.add(current_id)
        value_map[current_id] = value
        current_id += 1
    
    # Aoki's cards
    a_ids = set()
    for value in B:
        a_ids.add(current_id)
        value_map[current_id] = value
        current_id += 1
    
    # Table cards
    table_ids = set()
    for value in C:
        table_ids.add(current_id)
        value_map[current_id] = value
        current_id += 1
    
    @lru_cache(maxsize=None)
    def can_win(t_hand, a_hand, table, player):
        t_hand_set = set(t_hand)
        a_hand_set = set(a_hand)
        table_set = set(table)
        
        if player == 'T':
            if not t_hand_set:
                return False
            for card_id in t_hand_set:
                card_value = value_map[card_id]
                new_table = table_set.union({card_id})
                possible_take_backs = [c for c in table_set if value_map[c] < card_value]
                options = [None] + possible_take_backs
                for option in options:
                    if option is not None:
                        new_t_hand = t_hand_set - {card_id}
                        new_t_hand = new_t_hand.union({option})
                        new_table = new_table - {option}
                    else:
                        new_t_hand = t_hand_set - {card_id}
                        new_table = new_table
                    # Switch to Aoki's turn
                    if not can_win(frozenset(new_t_hand), a_hand, frozenset(new_table), 'A'):
                        return True
            return False
        else:
            if not a_hand_set:
                return False
            for card_id in a_hand_set:
                card_value = value_map[card_id]
                new_table = table_set.union({card_id})
                possible_take_backs = [c for c in table_set if value_map[c] < card_value]
                options = [None] + possible_take_backs
                for option in options:
                    if option is not None:
                        new_a_hand = a_hand_set - {card_id}
                        new_a_hand = new_a_hand.union({option})
                        new_table = new_table - {option}
                    else:
                        new_a_hand = a_hand_set - {card_id}
                        new_table = new_table
                    # Switch to Takahashi's turn
                    if not can_win(t_hand, frozenset(new_a_hand), frozenset(new_table), 'T'):
                        return True
            return False
    
    t_initial = frozenset(t_ids)
    a_initial = frozenset(a_ids)
    table_initial = frozenset(table_ids)
    
    if can_win(t_initial, a_initial, table_initial, 'T'):
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == "__main__":
    main()