def solve():
    import sys
    from collections import deque

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    S = input_data[1]
    T = input_data[2]

    # Quick check: if S and T differ in the number of black/white stones, impossible
    if S.count('B') != T.count('B') or S.count('W') != T.count('W'):
        print(-1)
        return

    # Initial and target states (N+2 cells, last two are empty)
    start = S + ".."
    goal = T + ".."

    # If already in final state
    if start == goal:
        print(0)
        return

    # BFS over all possible states
    visited = set([start])
    queue = [start]
    dist = 0

    while queue:
        dist += 1
        next_layer = []
        for state in queue:
            # Find the pair of empty cells (must be adjacent by problem statement)
            empty_pos = state.find("..")
            # Generate neighbors by moving any adjacent pair of stones into these empty slots
            for x in range(N+1):
                if state[x] != '.' and state[x+1] != '.':
                    # Move stones at x, x+1 to empty_pos, empty_pos+1
                    lst = list(state)
                    # Swap
                    lst[empty_pos], lst[empty_pos+1], lst[x], lst[x+1] = lst[x], lst[x+1], '.', '.'
                    new_state = "".join(lst)
                    if new_state not in visited:
                        if new_state == goal:
                            print(dist)
                            return
                        visited.add(new_state)
                        next_layer.append(new_state)
        queue = next_layer

    # If we exhaust all possibilities without reaching goal
    print(-1)

# Call solve()
solve()