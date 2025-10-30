import sys

def solve():
    N = int(sys.stdin.readline())
    N_points = 2 * N

    # partner[point_id] stores the other endpoint of the chord
    # Using a list of size 2N+1 for 1-based indexing of points.
    partner = [0] * (N_points + 1)
    for _ in range(N):
        A, B = map(int, sys.stdin.readline().split())
        partner[A] = B
        partner[B] = A

    # stack stores the 'far' end (larger endpoint) of currently open chords.
    # The top of the stack always represents the 'far' end of the most recently
    # opened chord that is still active.
    stack = []

    # Iterate through points on the circle in clockwise order.
    for p in range(1, N_points + 1):
        # If p is the starting point of a chord (i.e., its coordinate is smaller than its partner's)
        if p < partner[p]:
            stack.append(partner[p])
        # If p is the ending point of a chord (i.e., its coordinate is larger than its partner's)
        else:
            # We've reached an ending point 'p'. This chord started at partner[p].
            # For there to be no intersection with other currently open chords,
            # 'p' must be the ending point of the most recently opened chord.
            # This means 'p' should be the element at the top of the stack.
            
            # Check if stack is empty or if the top element is not 'p'.
            # If stack is empty, it implies an ending point without a corresponding
            # opening point, which for valid inputs means an intersection.
            # If stack[-1] != p, it means a chord (partner[p], p) is trying to close,
            # but another chord (starting earlier, ending at stack[-1]) is "inside" it
            # and still active, meaning they cross.
            if not stack or stack[-1] != p:
                sys.stdout.write("Yes
")
                return
            else:
                # The chord (partner[p], p) correctly closes without crossing others.
                # Pop its ending point from the stack.
                stack.pop()

    # If the loop completes without finding any intersections, print "No".
    sys.stdout.write("No
")

# Call the solve function to run the program.
solve()