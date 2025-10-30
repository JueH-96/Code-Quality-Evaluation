import sys

# Function to solve the problem
def solve():
    # Read the number of chords N from stdin
    N = int(sys.stdin.readline())
    
    # Create an array `partner` to store the connection information.
    # `partner[i]` will store the point connected to point `i` by a chord.
    # Initialize with 0s. The array size is 2*N + 1 to accommodate 1-based indexing up to 2N.
    partner = [0] * (2 * N + 1)
    
    # Read the N chords from stdin and populate the `partner` array.
    for _ in range(N):
        # Read the two endpoints A and B of a chord.
        # Using map(int, ...) is a standard way to parse space-separated integers.
        A, B = map(int, sys.stdin.readline().split())
        # Store the partner information: point A is connected to B, and B is connected to A.
        partner[A] = B
        partner[B] = A
    
    # Initialize an empty list to be used as a stack.
    # The stack will keep track of the "opening" endpoints of chords that are currently "active" (not yet closed).
    stack = []
    
    # Iterate through the points on the circle in clockwise order, from 1 to 2N.
    for P in range(1, 2 * N + 1):
        # Get the point Q connected to the current point P by a chord.
        Q = partner[P]
        
        # Check if P is the endpoint with the smaller index for its chord (P < Q).
        # If P < Q, this point P represents the start or "opening" of the chord (P, Q).
        if P < Q: 
            # Push the index P onto the stack. This marks the chord (P, Q) as active.
            stack.append(P)
        else: # P is the endpoint with the larger index (P > Q).
            # This point P represents the end or "closing" of the chord (Q, P).
            
            # According to the problem constraints and the logic, when we encounter a closing point P,
            # its corresponding opening point Q must have been pushed onto the stack earlier and should
            # ideally be at the top of the stack if there are no intersections involving this chord
            # and chords opened after Q.
            # The problem guarantees valid inputs, which implies the stack should not be empty here.
            # If the stack were empty, `stack.pop()` would raise an IndexError.
            
            # Pop the index of the most recently opened endpoint from the stack.
            last_opened = stack.pop()
            
            # Check if the popped endpoint index `last_opened` matches the partner `Q` of the current point `P`.
            # If `last_opened` is not equal to `Q`, it means that the chord we are trying to close, (Q, P),
            # is not the most recently opened active chord. The chord (last_opened, partner[last_opened])
            # was opened more recently (since `last_opened` was on top of the stack) but has not been closed yet.
            # This configuration implies an intersection between chord (Q, P) and chord (last_opened, partner[last_opened]).
            # The sequence of points encountered would be Q < last_opened < P < partner[last_opened], which satisfies the intersection condition.
            if last_opened != Q:
                # An intersection is detected. Print "Yes" and terminate the program.
                print("Yes")
                return

    # If the loop completes without returning 'Yes', it means that all chords were processed,
    # and every closing point P correctly matched the most recently opened endpoint Q from the stack.
    # This indicates that there are no intersecting chords.
    print("No")

# Call the solve function to execute the logic for reading input, processing, and printing output.
solve()