# YOUR CODE HERE
import sys

# Read the input string S from standard input.
S = sys.stdin.readline().strip()

# Use a state variable to track which segment (A, B, or C)
# of the A*B*C* pattern we are currently processing.
# State 0: In the A-segment phase (or before transitioning to B/C).
# State 1: In the B-segment phase (transitioned from A, cannot go back to A).
# State 2: In the C-segment phase (transitioned from A or B, cannot go back to A/B).
state = 0

# Iterate through each character in the input string.
for char in S:
    if state == 0:
        # Expecting A's, potentially transitioning to B or C.
        if char == 'A':
            # Correct: Stay in A-phase.
            pass
        elif char == 'B':
            # Transition to B-phase.
            state = 1
        elif char == 'C':
            # Transition directly to C-phase (empty B-segment).
            state = 2
        # According to problem constraints, the input string only contains 'A', 'B', or 'C'.
    elif state == 1:
        # Expecting B's, potentially transitioning to C.
        if char == 'A':
            # Invalid: Found 'A' after seeing 'B'.
            print("No")
            sys.exit()
        elif char == 'B':
            # Correct: Stay in B-phase.
            pass
        elif char == 'C':
            # Transition to C-phase.
            state = 2
        # According to problem constraints, the input string only contains 'A', 'B', or 'C'.
    elif state == 2:
        # Expecting only C's.
        if char == 'A' or char == 'B':
            # Invalid: Found 'A' or 'B' after seeing 'C'.
            print("No")
            sys.exit()
        elif char == 'C':
            # Correct: Stay in C-phase.
            pass
        # According to problem constraints, the input string only contains 'A', 'B', or 'C'.

# If the loop completes without exiting, it means the string matches the A*B*C* pattern.
print("Yes")