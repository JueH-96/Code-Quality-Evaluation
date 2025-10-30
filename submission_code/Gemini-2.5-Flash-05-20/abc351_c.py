import sys

def solve():
    # Read N, the total number of balls/operations
    N = int(sys.stdin.readline())

    # Read the exponents A_i for each ball.
    # We represent balls by their exponents directly, as 2^X + 2^X = 2^(X+1).
    A = list(map(int, sys.stdin.readline().split()))

    # The 'sequence_exponents' list will store the exponents of the balls
    # currently in the sequence. It acts as a stack: new balls are added
    # to the right end (using append), and merging operations always involve
    # the rightmost elements (using pop from the end).
    sequence_exponents = []

    # Iterate through each of the N balls, performing one operation per ball.
    for i in range(N):
        # Get the exponent of the current ball to be added.
        current_ball_exp = A[i]

        # Add the current ball to the right end of the sequence.
        sequence_exponents.append(current_ball_exp)

        # After adding a ball, initiate the merging process.
        # This process repeats as long as the conditions for merging are met.
        while True:
            # Step 1: If the sequence has one or fewer balls, no merging is possible.
            # End the current operation for this ball and move to the next A_i.
            if len(sequence_exponents) <= 1:
                break

            # Get the exponents of the rightmost and second rightmost balls.
            rightmost_exp = sequence_exponents[-1]
            second_rightmost_exp = sequence_exponents[-2]

            # Step 2: If the rightmost and second rightmost balls have different sizes (exponents),
            # no merging is possible. End the current operation.
            if rightmost_exp != second_rightmost_exp:
                break

            # Step 3: If the rightmost and second rightmost balls have the same size (exponent),
            # perform the merge operation.
            
            # Remove the two rightmost balls from the sequence.
            sequence_exponents.pop() # Removes the rightmost ball
            sequence_exponents.pop() # Removes the second rightmost ball

            # Add a new ball to the right end of the sequence.
            # Its size is the sum of the two removed balls.
            # Since 2^X + 2^X = 2 * 2^X = 2^(X+1), the new ball's exponent is X+1.
            merged_exp = rightmost_exp + 1 
            sequence_exponents.append(merged_exp)

            # The 'while True' loop implicitly handles "go back to step 1 and repeat".
            # After a merge, the conditions are re-evaluated to see if further merges are possible.

    # After all N operations (adding balls and performing merges) are complete,
    # the number of balls remaining in the sequence is simply the length of our list.
    print(len(sequence_exponents))

# Call the solve function to execute the program.
solve()