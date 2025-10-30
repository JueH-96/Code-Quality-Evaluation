def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    # Read N (number of parts) and Q (number of instructions)
    N = int(data[0])
    Q = int(data[1])
    # The initial configuration:
    # Left hand on part 1 and right hand on part 2.
    left = 1
    right = 2
    total_moves = 0
    idx = 2

    # Helper function to simulate moving from a starting part x to target t 
    # in a given direction ("cw" for clockwise, "acw" for anticlockwise)
    # while keeping the other hand (obstacle) fixed.
    # Returns the number of moves if the route is not blocked, or None if it is blocked.
    def simulate_path(x, t, obstacle, direction, N):
        count = 0
        pos = x
        while pos != t:
            if direction == "cw":
                next_pos = pos + 1 if pos < N else 1
            else:  # anticlockwise
                next_pos = pos - 1 if pos > 1 else N
            if next_pos == obstacle:  # cannot move into the part held by the other hand
                return None
            pos = next_pos
            count += 1
        return count

    # Process each instruction in order.
    for _ in range(Q):
        hand = data[idx]   # either "L" or "R"
        idx += 1
        target = int(data[idx])
        idx += 1
        
        if hand == "L":
            # When moving left hand, the right hand (at position 'right') is fixed.
            if left == target:
                moves_needed = 0
            else:
                # Simulate clockwise and anticlockwise paths.
                cw_moves = simulate_path(left, target, right, "cw", N)
                acw_moves = simulate_path(left, target, right, "acw", N)
                if cw_moves is None and acw_moves is None:
                    # According to the problem, this should not happen.
                    moves_needed = 10**9
                elif cw_moves is None:
                    moves_needed = acw_moves
                elif acw_moves is None:
                    moves_needed = cw_moves
                else:
                    moves_needed = min(cw_moves, acw_moves)
            total_moves += moves_needed
            left = target  # update left hand position
        else:  # hand == "R"
            # When moving right hand, the left hand (at position 'left') stays fixed.
            if right == target:
                moves_needed = 0
            else:
                cw_moves = simulate_path(right, target, left, "cw", N)
                acw_moves = simulate_path(right, target, left, "acw", N)
                if cw_moves is None and acw_moves is None:
                    moves_needed = 10**9
                elif cw_moves is None:
                    moves_needed = acw_moves
                elif acw_moves is None:
                    moves_needed = cw_moves
                else:
                    moves_needed = min(cw_moves, acw_moves)
            total_moves += moves_needed
            right = target  # update right hand position

    sys.stdout.write(str(total_moves))


if __name__ == '__main__':
    main()