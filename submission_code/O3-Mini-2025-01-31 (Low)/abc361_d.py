def main():
    import sys
    from collections import deque

    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    S = data[1].strip()
    T = data[2].strip()

    total = N + 2  # total number of cells

    # Represent state as a tuple of characters of length total.
    # The initial state has cells 1..N (0-indexed: 0..N-1) filled with S;
    # cells N+1 and N+2 are empty ('.').
    start = tuple(list(S) + ['.', '.'])
    # The goal is that cells 1..N (0-indexed: 0..N-1) are T and the last two cells are empty.
    goal = tuple(list(T) + ['.', '.'])

    # We'll use a BFS over states. The total number of states is moderate.
    # Each move: choose a pair of adjacent cells that both contain stones and
    # then choose a pair of adjacent cells that are empty; move the two stones (preserving order)
    # from the selected stone pair to the chosen empty pair.
    #
    # Note: In any state, because there are exactly N stones and two empties,
    # the empty cells always come as a pair in adjacent positions.
    #
    # However, they may not be contiguous in the order of stones until the final state.
    # We simulate the moves exactly as described.

    q = deque()
    q.append(start)
    dist = {start: 0}

    while q:
        state = q.popleft()
        d = dist[state]
        if state == goal:
            sys.stdout.write(str(d))
            return

        # Identify all valid source positions for moving a pair of adjacent stones.
        stone_pairs = []
        for i in range(total - 1):
            if state[i] != '.' and state[i + 1] != '.':
                stone_pairs.append(i)

        # Identify all valid target pairs (adjacent empty cells).
        empty_pairs = []
        for j in range(total - 1):
            if state[j] == '.' and state[j + 1] == '.':
                empty_pairs.append(j)

        # Try all moves. There are no additional restrictions because if a cell is 
        # occupied with a stone, it will not be empty. So the chosen target pair and source pair 
        # are naturally disjoint.
        for i in stone_pairs:
            for j in empty_pairs:
                # We create a new configuration by moving the two stones from positions i and i+1 
                # to positions j and j+1 respectively.
                newState = list(state)
                newState[j] = state[i]
                newState[j + 1] = state[i + 1]
                newState[i] = '.'
                newState[i + 1] = '.'
                new_state = tuple(newState)
                if new_state not in dist:
                    dist[new_state] = d + 1
                    q.append(new_state)

    # If we exhaust the search without reaching the goal, it's impossible.
    sys.stdout.write(str(-1))


if __name__ == "__main__":
    main()