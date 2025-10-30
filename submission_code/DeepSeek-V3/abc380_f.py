from functools import lru_cache

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    L = int(data[2])
    
    A = list(map(int, data[3:3+N]))
    B = list(map(int, data[3+N:3+N+M]))
    C = list(map(int, data[3+N+M:3+N+M+L]))
    
    # Since the order of cards in hand and on the table doesn't matter for the game outcome,
    # we can represent the state as the sorted lists of cards in each player's hand and on the table.
    # However, for simplicity, we can use the counts of cards and their values.
    
    # To represent the state, we can use a tuple of (sorted A, sorted B, sorted C)
    # But since the order doesn't matter, we can sort them and use the sorted lists.
    
    A_sorted = sorted(A)
    B_sorted = sorted(B)
    C_sorted = sorted(C)
    
    @lru_cache(maxsize=None)
    def can_win(takahashi_hand, aoki_hand, table):
        # Convert the tuples back to lists for easier manipulation
        takahashi_hand = list(takahashi_hand)
        aoki_hand = list(aoki_hand)
        table = list(table)
        
        # Takahashi's turn
        if not takahashi_hand:
            return False  # Takahashi cannot move, Aoki wins
        
        # Try all possible moves for Takahashi
        for i in range(len(takahashi_hand)):
            card = takahashi_hand[i]
            new_takahashi_hand = takahashi_hand[:i] + takahashi_hand[i+1:]
            new_table = table.copy()
            new_table.append(card)
            # Takahashi can take any card from the table that is less than the played card
            # So, we need to consider all possible subsets of cards to take
            # Since the number of cards is small, we can try all possible combinations
            # But for efficiency, we can precompute the possible cards to take
            possible_takes = [c for c in new_table if c < card]
            # Try all possible subsets of possible_takes
            # Since the number of cards is small, we can try all possible combinations
            # For each subset, we add the cards to Takahashi's hand and remove from the table
            # Then, we check if Aoki can win from the new state
            # If for any move, Aoki cannot win, then Takahashi can win
            # So, we need to find at least one move where Aoki cannot win
            # To handle this, we can generate all possible subsets of possible_takes
            # and for each subset, create the new state and check if Aoki can win
            # Since the number of subsets is 2^k where k is the number of possible_takes
            # and k is small (up to 12), this is manageable
            # So, we generate all subsets
            from itertools import combinations
            for r in range(len(possible_takes) + 1):
                for subset in combinations(possible_takes, r):
                    new_takahashi_hand_copy = new_takahashi_hand.copy()
                    new_table_copy = new_table.copy()
                    for c in subset:
                        new_takahashi_hand_copy.append(c)
                        new_table_copy.remove(c)
                    # Now, it's Aoki's turn
                    # We need to check if Aoki can win from this state
                    # So, we call can_win with the roles reversed
                    # Since it's Aoki's turn, we pass aoki_hand as the first argument
                    # and takahashi_hand_copy as the second
                    # and the table is new_table_copy
                    # So, the new state is (aoki_hand, new_takahashi_hand_copy, new_table_copy)
                    # But since the function is defined for Takahashi's turn, we need to swap the roles
                    # So, we need to define a helper function that swaps the roles
                    # Alternatively, we can define the function to take the current player as an argument
                    # But for simplicity, we can define a helper function
                    def aoki_can_win(aoki_hand, takahashi_hand, table):
                        # Aoki's turn
                        if not aoki_hand:
                            return False  # Aoki cannot move, Takahashi wins
                        # Try all possible moves for Aoki
                        for i in range(len(aoki_hand)):
                            card = aoki_hand[i]
                            new_aoki_hand = aoki_hand[:i] + aoki_hand[i+1:]
                            new_table = table.copy()
                            new_table.append(card)
                            possible_takes = [c for c in new_table if c < card]
                            from itertools import combinations
                            for r in range(len(possible_takes) + 1):
                                for subset in combinations(possible_takes, r):
                                    new_aoki_hand_copy = new_aoki_hand.copy()
                                    new_table_copy = new_table.copy()
                                    for c in subset:
                                        new_aoki_hand_copy.append(c)
                                        new_table_copy.remove(c)
                                    # Now, it's Takahashi's turn
                                    # We need to check if Takahashi can win from this state
                                    # So, we call can_win with the roles reversed
                                    if not can_win(takahashi_hand, new_aoki_hand_copy, tuple(sorted(new_table_copy))):
                                        return True
                        return False
                    if not aoki_can_win(aoki_hand, new_takahashi_hand_copy, tuple(sorted(new_table_copy))):
                        return True
        return False
    
    # Initialize the game state
    takahashi_hand = tuple(sorted(A))
    aoki_hand = tuple(sorted(B))
    table = tuple(sorted(C))
    
    if can_win(takahashi_hand, aoki_hand, table):
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == "__main__":
    main()