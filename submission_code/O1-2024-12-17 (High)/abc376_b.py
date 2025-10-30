def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, Q = map(int, data[:2])

    # The initial positions of the hands
    left_hand = 1
    right_hand = 2

    # Helper functions to compute distance ignoring a blocked node
    def move_cw(start, target, block, n):
        """Return number of steps moving clockwise from start to target 
           without stepping on 'block', or None if impossible."""
        steps = 0
        current = start
        while True:
            current = (current % n) + 1
            steps += 1
            if current == block:
                return None  # Blocked
            if current == target:
                return steps

    def move_ccw(start, target, block, n):
        """Return number of steps moving counterclockwise from start to target 
           without stepping on 'block', or None if impossible."""
        steps = 0
        current = start
        while True:
            current = (current - 2) % n + 1
            steps += 1
            if current == block:
                return None  # Blocked
            if current == target:
                return steps

    def dist(a, t, b, n):
        """Minimum steps to move from a to t in a ring of size n,
           forbidding stepping on b."""
        if a == t:
            return 0
        cw = move_cw(a, t, b, n)
        ccw = move_ccw(a, t, b, n)
        if cw is not None and ccw is not None:
            return min(cw, ccw)
        elif cw is not None:
            return cw
        elif ccw is not None:
            return ccw
        # Problem statement guarantees a feasible path, so no else case needed
        return 0

    total_operations = 0
    idx = 2
    for _ in range(Q):
        hand = data[idx]
        idx += 1
        target = int(data[idx])
        idx += 1

        if hand == 'L':
            # Move the left hand from left_hand to target (block is right_hand)
            total_operations += dist(left_hand, target, right_hand, N)
            left_hand = target
        else:  # hand == 'R'
            # Move the right hand from right_hand to target (block is left_hand)
            total_operations += dist(right_hand, target, left_hand, N)
            right_hand = target

    print(total_operations)

# Do not forget to call main()!
if __name__ == "__main__":
    main()