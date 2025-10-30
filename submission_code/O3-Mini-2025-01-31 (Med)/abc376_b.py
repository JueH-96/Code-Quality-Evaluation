def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    Q = int(next(it))
    
    # Helper function to compute the minimum operations from a to b on a ring 
    # of N parts, moving only step by step (adjacent moves) and ensuring that
    # the forbidden part (held by the other hand) is never passed through.
    def min_steps(a, b, forbidden, N):
        if a == b:
            return 0
        
        best = None
        
        # Try the clockwise direction:
        steps = 0
        cur = a
        valid = True
        while True:
            cur = cur + 1
            if cur > N:
                cur = 1
            steps += 1
            if cur == forbidden:
                valid = False
                break
            if cur == b:
                break
        if valid:
            best = steps
        
        # Try the anticlockwise direction:
        steps = 0
        cur = a
        valid2 = True
        while True:
            cur = cur - 1
            if cur < 1:
                cur = N
            steps += 1
            if cur == forbidden:
                valid2 = False
                break
            if cur == b:
                break
        if valid2:
            if best is None or steps < best:
                best = steps
        
        # One valid path must exist (per problem guarantees)
        return best
    
    # Initial positions: left hand on part 1, right hand on part 2.
    left, right = 1, 2
    total_moves = 0
    
    for _ in range(Q):
        hand = next(it)
        target = int(next(it))
        if hand == "L":
            # Move left hand from current position (left) to target.
            # The right hand (at 'right') is forbidden.
            move_count = min_steps(left, target, right, N)
            total_moves += move_count
            left = target
        else:  # hand == "R"
            # Move right hand from current position (right) to target.
            # The left hand (at 'left') is forbidden.
            move_count = min_steps(right, target, left, N)
            total_moves += move_count
            right = target

    sys.stdout.write(str(total_moves))

if __name__ == '__main__':
    main()