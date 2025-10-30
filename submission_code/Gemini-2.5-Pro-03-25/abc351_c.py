# YOUR CODE HERE
import sys

# Function to solve the problem
def solve():
    """
    Reads input, simulates the ball adding and merging process,
    and prints the final number of balls in the sequence.
    """
    # Read the number of balls N from standard input
    n = int(sys.stdin.readline())
    
    # Read the list of exponents A from standard input
    # Each A_i corresponds to the size 2^(A_i) of the i-th ball
    a = list(map(int, sys.stdin.readline().split()))

    # Initialize an empty list to represent the sequence of balls.
    # We will store the exponents in this list, as the actual sizes 2^A_i
    # are only needed to check equality, which is equivalent to checking exponent equality.
    # Merging two balls of size 2^k results in a ball of size 2^k + 2^k = 2 * 2^k = 2^(k+1).
    # So, merging involves incrementing the exponent by 1.
    sequence = []

    # Iterate through each exponent provided in the input list A
    for x in a:
        # Add the current ball's exponent to the right end of the sequence
        sequence.append(x)
        
        # After adding a ball, repeatedly check if the last two balls can be merged.
        # The merge process continues as long as there are at least two balls in the sequence
        # and the last two balls have the same size (which means they have the same exponent).
        while len(sequence) >= 2:
            # Check if the exponent of the rightmost ball (sequence[-1])
            # is equal to the exponent of the second rightmost ball (sequence[-2])
            if sequence[-1] == sequence[-2]:
                # If they are equal, these two balls merge.
                
                # Remove the rightmost ball's exponent. The pop() method removes and returns the last element.
                last_exp = sequence.pop()
                # Remove the second rightmost ball's exponent.
                sequence.pop() 
                
                # Calculate the exponent of the new ball formed by the merge.
                # The new ball's size is the sum of the two merged balls: 2^k + 2^k = 2^(k+1).
                # So the new exponent is the old exponent + 1.
                new_exp = last_exp + 1
                
                # Add the new ball's exponent to the right end of the sequence.
                sequence.append(new_exp)
                
                # After merging, the newly added ball might be able to merge with the ball now to its left.
                # The while loop condition (len(sequence) >= 2) ensures we re-check this possibility
                # in the next iteration of the while loop.
            else:
                # If the last two balls have different sizes (different exponents),
                # no further merging is possible with the current sequence configuration.
                # Break out of the inner while loop. The outer loop will proceed to the next ball.
                break

    # After processing all N balls and performing all associated merges,
    # the final number of balls remaining in the sequence is simply the current length of the list.
    # Print the length of the final sequence to standard output.
    print(len(sequence))

# Execute the solve function to run the program logic based on standard input/output
solve()

# END OF YOUR CODE HERE