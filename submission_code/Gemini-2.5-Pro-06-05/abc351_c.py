import sys

def solve():
    """
    Reads the problem input, simulates the ball merging process using a stack,
    and prints the final number of balls.
    """
    # Read N, the number of operations. According to constraints, N >= 1.
    try:
        N = int(sys.stdin.readline())
    except (ValueError, IndexError):
        # This case should not be reached given the problem constraints.
        # Handle gracefully for robustness.
        print(0)
        return

    # If N is 0, no balls are added.
    if N == 0:
        print(0)
        return

    # Read the list of exponents A.
    A = list(map(int, sys.stdin.readline().split()))

    # `sequence` will store the exponents of the balls currently in the sequence.
    # We use a list and treat it as a stack.
    sequence = []

    # Process each given ball.
    for exponent in A:
        # Add the new ball to the right end of the sequence.
        sequence.append(exponent)
        
        # Repeatedly merge the two rightmost balls if they have the same size.
        # This is checked by comparing their exponents.
        while len(sequence) >= 2 and sequence[-1] == sequence[-2]:
            # Pop the exponent of the rightmost ball.
            last_exponent = sequence.pop()
            
            # Pop the exponent of the new rightmost ball (which was the second rightmost).
            sequence.pop()
            
            # Add a new ball representing the merged pair.
            # Its size is 2 * 2^E = 2^(E+1), so its exponent is E+1.
            sequence.append(last_exponent + 1)
            
    # The final answer is the number of balls remaining in the sequence.
    print(len(sequence))

solve()