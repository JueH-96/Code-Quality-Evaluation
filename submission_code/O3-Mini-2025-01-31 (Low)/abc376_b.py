def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    Q = int(next(it))
    
    # Start with left hand at part 1 and right hand at part 2.
    left = 1
    right = 2
    total_operations = 0

    # Helper: generate the path (list of parts visited) from start to target in a specified direction.
    # It excludes the starting part and includes the target.
    def get_path(start, target, clockwise=True):
        path = []
        current = start
        while True:
            if clockwise:
                # Move to next part in clockwise order (wrap around)
                current = current + 1 if current < N else 1
            else:
                # Move anticlockwise (wrap around)
                current = current - 1 if current > 1 else N
            path.append(current)
            if current == target:
                break
        return path

    for _ in range(Q):
        # Read the instruction: hand to move and the target part.
        hand = next(it)
        target = int(next(it))
        # For the instruction, the other hand remains still.
        if hand == 'L':
            current_position = left
            other_position = right
        else:
            current_position = right
            other_position = left

        # Compute two possible paths: clockwise and anticlockwise.
        cw_path = get_path(current_position, target, clockwise=True)
        ccw_path = get_path(current_position, target, clockwise=False)

        # We cannot choose a path that would require moving onto the part
        # currently held by the other hand.
        cw_possible = (other_position not in cw_path)
        ccw_possible = (other_position not in ccw_path)

        # Among the valid directions, choose the one with fewer moves.
        possible_moves = []
        if cw_possible:
            possible_moves.append(len(cw_path))
        if ccw_possible:
            possible_moves.append(len(ccw_path))
        
        # There is guaranteed to be at least one valid option.
        moves = min(possible_moves)
        total_operations += moves

        # Update the hand position that moved.
        if hand == 'L':
            left = target
        else:
            right = target

    sys.stdout.write(str(total_operations))


if __name__ == '__main__':
    main()