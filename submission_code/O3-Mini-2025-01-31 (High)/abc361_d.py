# YOUR CODE HERE
def main():
    import sys
    from collections import deque

    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    S = data[1].strip()
    T = data[2].strip()
    L = N + 2
    # Initial board: cells 0..N-1 contain stones given by S, and cells N and N+1 are empty.
    init = S + ".."
    # Target board: cells 0..N-1 should have stones given by T, and cells N and N+1 are empty.
    target = T + ".."

    # In every valid move exactly two consecutive cells are empty.
    # The allowed operation is to select any pair of adjacent cells (i, i+1) that are both
    # occupied by stones, then “swap” them with the unique two‐cell gap (in which the cells are empty).
    # That is, if the gap is at positions (g, g+1), then the stones at (i, i+1) will go into (g, g+1)
    # and (i, i+1) become empty.
    
    dq = deque()
    dq.append(init)
    visited = {init: 0}
    
    while dq:
        state = dq.popleft()
        d = visited[state]
        if state == target:
            sys.stdout.write(str(d))
            return
        
        # Find the gap. Our invariant guarantees there is exactly one pair of consecutive dots.
        gap = None
        for i in range(L - 1):
            if state[i] == '.' and state[i+1] == '.':
                gap = i
                break
        
        # For every candidate pair of adjacent stones in the board, perform the swap with the gap.
        for i in range(L - 1):
            if state[i] != '.' and state[i+1] != '.':
                new_list = list(state)
                # Move the stones from positions i and i+1 to the current gap positions.
                new_list[gap]   = state[i]
                new_list[gap+1] = state[i+1]
                # Their original positions become empty.
                new_list[i]   = '.'
                new_list[i+1] = '.'
                new_state = "".join(new_list)
                if new_state not in visited:
                    visited[new_state] = d + 1
                    dq.append(new_state)
                    
    sys.stdout.write("-1")

if __name__ == "__main__":
    main()