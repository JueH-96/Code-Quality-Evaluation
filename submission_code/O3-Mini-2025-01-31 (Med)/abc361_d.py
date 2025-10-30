def main():
    import sys
    from collections import deque

    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    S = data[1].strip()
    T = data[2].strip()

    # The board has N+2 cells.
    # Initially, cells 0..(N-1) have stones with colors from S,
    # and cells N and N+1 are empty (represented by '.').
    # The target state is cells 0..(N-1) filled with stones in order T,
    # and cells N and N+1 empty.
    start = S + ".."
    target = T + ".."
    if start == target:
        sys.stdout.write("0")
        return

    board_len = N + 2
    # BFS using states represented as strings of length board_len.
    # Note: An invariant is that the two empty cells are consecutive.
    # Thus the allowed move is:
    #    • Choose any adjacent pair (cells x and x+1) that both are stones.
    #    • Move them to the unique empty pair (cells e and e+1) [found in the state],
    #      preserving the order, and leave the original positions empty.
    #
    # That is, if our state is represented as a list state[0...board_len-1],
    # and if empty pair is at positions (e, e+1), then for any adjacent stone pair
    # at positions (x, x+1), we can construct a new state as follows:
    #    new_state[e] = state[x]
    #    new_state[e+1] = state[x+1]
    #    new_state[x] = '.'
    #    new_state[x+1] = '.'
    #    Other positions remain the same.
    dq = deque()
    dq.append(start)
    visited = {start: 0}

    while dq:
        state = dq.popleft()
        steps = visited[state]

        # Find the index of the empty adjacent pair (guaranteed to exist in a valid state).
        empty_index = None
        for i in range(board_len - 1):
            if state[i] == '.' and state[i+1] == '.':
                empty_index = i
                break
        # Process every possible move: choose every adjacent stone pair.
        for x in range(board_len - 1):
            # Check if positions x and x+1 contain stones (not empty).
            if state[x] != '.' and state[x+1] != '.':
                lst = list(state)
                # Move stones from positions x, x+1 to the empty pair positions.
                lst[empty_index] = state[x]
                lst[empty_index+1] = state[x+1]
                # The original positions become empty.
                lst[x] = '.'
                lst[x+1] = '.'
                new_state = "".join(lst)
                if new_state not in visited:
                    visited[new_state] = steps + 1
                    if new_state == target:
                        sys.stdout.write(str(steps+1))
                        return
                    dq.append(new_state)
    # If target state is unreachable.
    sys.stdout.write("-1")

if __name__ == '__main__':
    main()