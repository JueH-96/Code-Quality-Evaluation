def solve():
    N = int(input())
    cards = []
    for _ in range(N):
        A, B = map(int, input().split())
        cards.append((A, B))
    
    # Memoization cache
    memo = {}
    
    def is_winning(state_cards):
        """
        Determines if the current player has a winning strategy from this state.
        A state is winning if there exists a move leading to a losing state for the opponent.
        """
        if state_cards in memo:
            return memo[state_cards]
        
        # Try all possible pairs of cards to remove
        for i in range(len(state_cards)):
            for j in range(i + 1, len(state_cards)):
                # Check if cards can be paired (same front or same back)
                if state_cards[i][0] == state_cards[j][0] or state_cards[i][1] == state_cards[j][1]:
                    # Create next state by removing the pair of cards
                    next_state_cards = tuple(state_cards[k] for k in range(len(state_cards)) if k != i and k != j)
                    # If opponent loses from next state, current player wins
                    if not is_winning(next_state_cards):
                        memo[state_cards] = True
                        return True
        
        # If no move leads to a winning position, the current player loses
        memo[state_cards] = False
        return False
    
    # Start with all cards and determine the winner
    result = is_winning(tuple(cards))
    return "Takahashi" if result else "Aoki"

print(solve())