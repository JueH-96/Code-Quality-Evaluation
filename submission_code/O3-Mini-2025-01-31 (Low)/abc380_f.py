def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    L = int(next(it))
    total = N + M + L

    # card values: first N are Takahashi's, next M are Aoki's, last L are table's.
    cardValues = []
    for _ in range(N):
        cardValues.append(int(next(it)))
    for _ in range(M):
        cardValues.append(int(next(it)))
    for _ in range(L):
        cardValues.append(int(next(it)))
    
    # initial state:
    # state[i] = 0 if card i is in Takahashi's hand, 1 if in Aoki's hand, 2 if on the table.
    state = []
    for i in range(N):
        state.append(0)
    for i in range(M):
        state.append(1)
    for i in range(L):
        state.append(2)
    
    # We'll encode state as an integer in base 3 for memoization because total <= 12.
    def encode(state):
        code = 0
        base = 1
        for s in state:
            code += s * base
            base *= 3
        return code

    # dp_memo[turn][encoded_state] = bool result for current player's turn (0 for Takahashi, 1 for Aoki)
    dp_memo = [dict(), dict()]
    
    # Launch recursive DFS minimax search.
    # turn: 0 for Takahashi, 1 for Aoki.
    def dfs(state, turn):
        key = encode(state)
        if key in dp_memo[turn]:
            return dp_memo[turn][key]
        # If current player's hand is empty, then they lose.
        has_move = any(s == turn for s in state)
        if not has_move:
            dp_memo[turn][key] = False
            return False

        # try every possible move: choose a card i in current hand.
        for i in range(total):
            if state[i] == turn:
                orig_val = state[i]
                # play card i: move from hand to table.
                state[i] = 2
                # In the move, we have the option of not picking up any card.
                # Get next state with turn switched.
                if not dfs(state, 1 - turn):
                    state[i] = orig_val
                    dp_memo[turn][key] = True
                    return True
                # Also, try picking up one card j from table (other than i) with cardValues[j] < cardValues[i].
                for j in range(total):
                    if j == i:
                        continue
                    if state[j] == 2 and cardValues[j] < cardValues[i]:
                        orig_j = state[j]
                        state[j] = turn  # pick it up into current player's hand
                        if not dfs(state, 1 - turn):
                            state[j] = orig_j
                            state[i] = orig_val
                            dp_memo[turn][key] = True
                            return True
                        state[j] = orig_j
                # Backtrack
                state[i] = orig_val
        dp_memo[turn][key] = False
        return False

    winner = "Takahashi" if dfs(state, 0) else "Aoki"
    sys.stdout.write(winner)

if __name__=="__main__":
    main()