def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    L = int(next(it))
    # Takahashi's cards
    takahashi_cards = [int(next(it)) for _ in range(N)]
    # Aoki's cards
    aoki_cards = [int(next(it)) for _ in range(M)]
    # Table cards
    table_cards = [int(next(it)) for _ in range(L)]
    
    # We now build a single list of card values.
    # We assign a fixed id to each card:
    #   Cards 0 .. N-1 are Takahashi's (player 1)
    #   Cards N .. N+M-1 are Aoki's (player 2)
    #   Cards N+M .. N+M+L-1 are on the table (location 0)
    card_vals = takahashi_cards + aoki_cards + table_cards
    total = len(card_vals)
    
    # We define the location for each card:
    #   location 1: Takahashi’s hand
    #   location 2: Aoki’s hand
    #   location 0: on the table
    loc_init = [None] * total
    for i in range(total):
        if i < N:
            loc_init[i] = 1  # Takahashi
        elif i < N + M:
            loc_init[i] = 2  # Aoki
        else:
            loc_init[i] = 0  # Table
    init_state = tuple(loc_init)
    
    # We use recursion with memoization.
    # The state will be a tuple (turn, locations)
    # where turn = 1 (Takahashi) or 2 (Aoki)
    # and locations is a tuple of length total with entries in {0,1,2}.
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dfs(turn, state):
        # First, if the current player has no card in hand, he loses.
        has_card = False
        for i in range(total):
            if state[i] == turn:
                has_card = True
                break
        if not has_card:
            return False
        
        # Try every possible move.
        # A move: choose one card in your hand (state[i] == turn) to play.
        # When you play card i, you:
        #   1) Remove it from your hand (set its location to 0, i.e. table).
        #   2) Optionally pick exactly one card from the table
        #      that has a number less than card_vals[i] (if any exist).
        #      You may also choose not to pick any card.
        for i in range(total):
            if state[i] != turn:
                continue
            # Playing card i.
            new_state = list(state)
            new_state[i] = 0  # card i is now on the table.
            
            # Option 1: Do NOT pick any extra card.
            next_state = tuple(new_state)
            # Turn switches to the opponent.
            if not dfs(3 - turn, next_state):
                return True

            # Option 2: For every card on the table with a lower number than card i, take that card.
            for j in range(total):
                if new_state[j] == 0 and card_vals[j] < card_vals[i]:
                    new_state2 = new_state.copy()
                    new_state2[j] = turn  # add card j to your hand.
                    next_state2 = tuple(new_state2)
                    if not dfs(3 - turn, next_state2):
                        return True
        # If no move gives a win, the current position is losing.
        return False

    if dfs(1, init_state):
        sys.stdout.write("Takahashi")
    else:
        sys.stdout.write("Aoki")

if __name__ == '__main__':
    main()