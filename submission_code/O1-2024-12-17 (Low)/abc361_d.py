from collections import deque

def solve_puzzle(N, S, T):
    """
    Returns the minimum number of operations needed to transform
    S (plus two empty cells at the end) into T (plus two empty cells at the end)
    under the puzzle rules, or -1 if it is impossible.
    """

    # Initial and goal configurations (with 2 empty cells at the end)
    initial = S + ".."
    goal_prefix = T  # We only care that the first N match T and last two are '.'

    # Quick check: if they are the same already, no moves are required
    if initial[:N] == goal_prefix:
        return 0

    # Each state will be stored as (configuration_string, empty_index),
    # where empty_index is the left index of the pair of empty cells in 0-based indexing.
    # This helps avoid scanning for empties every time.
    # The configuration is always length N+2, exactly two '.' side by side.
    # We'll find the empty index in the initial state (should be N in 0-based).
    empty_index = N  # Because initial has empties at positions N, N+1 (0-based)
    start_state = (initial, empty_index)

    visited = set([start_state])
    queue = deque([(start_state, 0)])  # (state, distance)

    while queue:
        (curr_config, eidx), dist = queue.popleft()

        # Try all possible pairs (x, x+1) that contain stones
        for x in range(N+1):  # x goes from 0 to N (inclusive) in 0-based
            if curr_config[x] != '.' and curr_config[x+1] != '.':
                # Construct new configuration by moving the pair from x,x+1 to eidx,eidx+1
                # We'll do this by swapping those characters with the '.' pair at eidx,eidx+1
                if x == eidx:
                    # That means the same place, no change
                    continue

                arr_list = list(curr_config)
                # Stones to move
                stone_left, stone_right = arr_list[x], arr_list[x+1]
                # Swap
                arr_list[eidx], arr_list[eidx+1] = stone_left, stone_right
                arr_list[x], arr_list[x+1] = '.', '.'
                new_config = ''.join(arr_list)
                new_eidx = x  # The empties moved to x,x+1 now

                new_state = (new_config, new_eidx)
                if new_state not in visited:
                    visited.add(new_state)
                    new_dist = dist + 1
                    # Check if this meets the goal (first N cells match T, last two are '.')
                    if new_config[:N] == goal_prefix:
                        return new_dist
                    queue.append((new_state, new_dist))

    # If not found
    return -1

def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]
    T = data[2]

    ans = solve_puzzle(N, S, T)
    print(ans)

# Do not forget to call main()
if __name__ == "__main__":
    main()