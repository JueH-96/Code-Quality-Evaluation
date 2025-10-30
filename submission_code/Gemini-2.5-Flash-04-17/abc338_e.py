import sys

def solve():
    # Read N, the number of chords.
    # N is between 2 and 2 * 10^5.
    N = int(sys.stdin.readline())

    # Create a mapping from each point number (1 to 2N) to its partner point number.
    # The problem guarantees that A_1,...,A_N, B_1,...,B_N are all distinct and cover 1 to 2N.
    # This means each point from 1 to 2N is an endpoint of exactly one chord.
    # Using a list for 1-based indexing, size 2N + 1. Initialize with 0 or None.
    partner = [0] * (2 * N + 1)
    for _ in range(N):
        # Read the endpoints of each chord (A_i, B_i).
        # A_i and B_i are between 1 and 2N.
        u, v = map(int, sys.stdin.readline().split())
        # Store the partnership in the array. Point u is partnered with v, and v with u.
        partner[u] = v
        partner[v] = u

    # Use a stack to keep track of the starting points of chords encountered so far
    # that have not yet been closed by their corresponding ending points.
    # The stack stores point numbers. A point is pushed onto the stack when we encounter it
    # and it is the first of its chord's endpoints in the clockwise sweep (relative to the
    # current state of the stack).
    stack = []
    # Use a set to allow efficient checking if a point is currently present in the stack.
    # This helps determine if a point is the closing endpoint of a chord whose starting
    # point is deeper in the stack (indicating an intersection). Set membership checking is O(1) on average.
    in_stack = set()

    # Iterate through points on the circle in clockwise order, from 1 to 2N.
    # These points represent the positions on the circle as we sweep around.
    # Each point p will be an endpoint of exactly one chord.
    for p in range(1, 2 * N + 1):
        # 'p' is the current point on the circle we are visiting in clockwise order.
        # We need to determine if 'p' is the starting or ending point of a chord,
        # relative to the chords currently "open" (represented by the stack).

        # Check if the stack is currently empty.
        if not stack:
            # If the stack is empty, the current point 'p' must be the first endpoint encountered
            # in this segment of the sweep. It signifies the start of a new chord.
            # Push this point onto the stack as an "open" starting point.
            # Add the point to the set tracking elements in the stack.
            stack.append(p)
            in_stack.add(p)
        else:
            # If the stack is not empty, the point at the top (`stack[-1]`) is the starting point
            # of the most recently opened chord that is still waiting for its closing endpoint.
            top_of_stack = stack[-1]

            # Check if the current point 'p' is the partner of the point at the top of the stack.
            # The partner of p is partner[p].
            # If partner[p] == top_of_stack, it means 'p' is the closing endpoint for the chord
            # that started at 'top_of_stack'. This indicates a properly nested or adjacent chord pair
            # relative to the most recently opened chord.
            # In a non-intersecting configuration, when we encounter an endpoint 'p', if its partner
            # was the point at the top of the stack, this chord (top_of_stack, p) is properly closing.
            if partner[p] == top_of_stack:
                # The chord (top_of_stack, p) is closed. Pop the starting point from the stack.
                # Also remove the point from the set tracking elements in the stack.
                stack.pop()
                in_stack.remove(top_of_stack)
            else:
                # If 'p' is not the partner of the top of the stack, there are two possibilities:
                # 1. 'p' is the starting point of a *new* chord (p, partner[p]) whose partner hasn't been seen yet.
                #    In this case, partner[p] would not be in the `in_stack` set.
                # 2. 'p' is the ending point of a chord (q, p) where 'q' is a starting point encountered *earlier*
                #    than 'top_of_stack' and is still in the stack (i.e., q is deeper in the stack).
                #    In this case, partner[p] (which is q) would be in the `in_stack` set. This scenario
                #    implies an intersection.

                # Check if the partner of the current point 'p' is present anywhere in the stack.
                # We use the 'in_stack' set for efficient O(1) average time lookup.
                if partner[p] in in_stack:
                    # If partner[p] is in the stack, and it's not the top (because we are in the 'else' block),
                    # it means partner[p] is a point 'q' that was pushed earlier than 'top_of_stack'.
                    # The stack contains ..., q, ..., top_of_stack. The points were encountered in the sweep
                    # in order ..., q, ..., top_of_stack, ..., p.
                    # The chord (q, p) is closing now. The chord (top_of_stack, partner[top_of_stack]) is open.
                    # Since top_of_stack is still in the stack, its partner (partner[top_of_stack]) must
                    # be a point that appears *after* 'p' in the clockwise sweep.
                    # This sequence of endpoints ..., q, ..., top_of_stack, ..., p, ..., partner[top_of_stack], ...
                    # visually demonstrates that chord (q, p) and chord (top_of_stack, partner[top_of_stack]) intersect.
                    print("Yes")
                    # Found an intersection, we can immediately terminate the function.
                    return

                else:
                    # If partner[p] is not in the stack at all, then 'p' must be the first endpoint
                    # encountered for the chord (p, partner[p]) in the current processing context.
                    # This chord is now considered "open". Push 'p' onto the stack to mark this.
                    # Add 'p' to the set tracking elements currently in the stack.
                    stack.append(p)
                    in_stack.add(p)

    # If the loop completes without finding any intersection (i.e., without executing the 'return' statement),
    # it means all chords could be processed in a way that respects the nesting property, or forms
    # non-intersecting configurations.
    # Print "No".
    # Note: Given the problem constraints (all 1..2N are distinct endpoints of N chords), if there are no
    # intersections, the stack must be empty at the end of the sweep. An empty stack implies no open chords
    # remain, consistent with no intersections.
    print("No")

# Execute the solve function to run the program.
solve()