def main():
    import sys
    sys.setrecursionlimit(10**7)
    
    data = sys.stdin.read().strip().split()
    N, M, L = map(int, data[:3])
    idx = 3
    A = list(map(int, data[idx:idx+N]))
    idx += N
    B = list(map(int, data[idx:idx+M]))
    idx += M
    C = list(map(int, data[idx:idx+L]))
    
    # We'll use a memo dictionary to store (handT, handA, table, turn) -> bool
    # True means the current player (turn) can force a win from that state.
    memo = {}
    
    # Convert lists to sorted tuples for canonical representation
    A.sort()
    B.sort()
    C.sort()
    
    # turn = 0 => Takahashi, turn = 1 => Aoki
    def can_win(t_hand, a_hand, table, turn):
        """
        Return True if the current player (turn) to move can force a win,
        False otherwise.
        """
        state = (t_hand, a_hand, table, turn)
        if state in memo:
            return memo[state]
        
        # Current player's hand
        if turn == 0:
            # Takahashi's hand
            cur_hand = t_hand
            opp_hand = a_hand
        else:
            # Aoki's hand
            cur_hand = a_hand
            opp_hand = t_hand
        
        # If current player's hand is empty, they cannot make a move => lose
        if not cur_hand:
            memo[state] = False
            return False
        
        # Try all possible cards in current player's hand
        # For each card, place it on the table, then optionally pick up
        # one smaller card from the table.
        
        cur_hand_list = list(cur_hand)
        table_list = list(table)
        
        for i, card in enumerate(cur_hand_list):
            # Remove card from current player's hand
            new_hand_list = cur_hand_list[:i] + cur_hand_list[i+1:]
            
            # Place the chosen card on the table
            new_table_list = table_list + [card]
            
            # We'll gather all possible picks from the table that are < card
            # This includes the option of not picking up any card.
            # But note that table now includes the newly placed card - which won't be less.
            
            # We'll handle picking from the table as it stood before we placed
            # the new card, because the newly added card can't be smaller than itself.
            # So we check table_list (WITHOUT the newly added card).
            
            smaller_indices = [idx_tbl for idx_tbl, tbl_card in enumerate(table_list) if tbl_card < card]
            
            # We also consider not picking anything
            pick_options = [None] + smaller_indices
            
            for pick_idx in pick_options:
                # Build new states
                if pick_idx is None:
                    # Not picking anything
                    pick_hand_list = new_hand_list
                    pick_table_list = new_table_list
                else:
                    # Pick table_list[pick_idx]
                    pick_card = table_list[pick_idx]
                    # Add pick_card to the current player's hand
                    pick_hand_list = new_hand_list + [pick_card]
                    # Remove that card from the table_list
                    # but note that we've already appended "card" to the new_table_list,
                    # so we want to remove one occurrence of pick_card from new_table_list (the old portion).
                    
                    # new_table_list = table_list + [card]
                    # We only remove the card from the old portion of the table (table_list)
                    # because we want to remove the index pick_idx from table_list specifically.
                    
                    # Easiest is to do it carefully from scratch:
                    pick_table_list = []
                    used_index = False
                    for idx_tbl, tbl_val in enumerate(table_list):
                        if idx_tbl == pick_idx and not used_index:
                            used_index = True
                        else:
                            pick_table_list.append(tbl_val)
                    # now add the newly placed card
                    pick_table_list.append(card)
                
                # Sort the new hand if we want canonical representation
                # Then decide whose hand is which next time
                pick_hand_list.sort()
                pick_table_list.sort()
                
                if turn == 0:
                    nxt_t_hand = tuple(pick_hand_list)
                    nxt_a_hand = a_hand
                else:
                    nxt_t_hand = t_hand
                    nxt_a_hand = tuple(pick_hand_list)
                
                nxt_table = tuple(pick_table_list)
                nxt_turn = 1 - turn
                
                if not can_win(nxt_t_hand, nxt_a_hand, nxt_table, nxt_turn):
                    # We found a move leading the opponent to lose => we win
                    memo[state] = True
                    return True
        
        # If no move leads to an opponent losing position => we lose
        memo[state] = False
        return False
    
    t_hand = tuple(A)
    a_hand = tuple(B)
    table = tuple(C)
    
    # If can_win(...) is True at the start (Takahashi to move), print Takahashi, else Aoki
    if can_win(t_hand, a_hand, table, 0):
        print("Takahashi")
    else:
        print("Aoki")

# Don't forget to call main()
if __name__ == "__main__":
    main()