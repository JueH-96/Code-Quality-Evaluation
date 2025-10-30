import sys

def solve():
    """
    This function reads an integer N from standard input, finds the largest
    palindromic cube number not greater than N, and prints it to standard output.
    """
    
    # Read the input integer N from standard input.
    # The problem constraints guarantee N is a positive integer up to 10^18.
    try:
        N = int(sys.stdin.readline())
    except (ValueError, IndexError):
        # This case should not occur based on problem constraints.
        return

    # Initialize the answer. 1 is the smallest palindromic cube (1^3 = 1).
    # Since N is a positive integer (N >= 1), 1 is always a valid candidate.
    ans = 1

    # Iterate through integers i, starting from 2, to find their cubes.
    # The cube root of a number up to 10^18 will be at most 10^6.
    # This loop will run a feasible number of times (approx. 10^6 iterations).
    i = 2
    while True:
        # Calculate the cube of i. Python's arbitrary-precision integers
        # handle large numbers, so overflow is not an issue.
        cube = i * i * i
        
        # If the cube exceeds N, we have passed all possible candidates.
        # The last found palindromic cube is the answer, so we can stop.
        if cube > N:
            break
            
        # Check if the cube is a palindrome. This is done by converting the
        # number to a string and checking if the string is equal to its reverse.
        s_cube = str(cube)
        if s_cube == s_cube[::-1]:
            # If it's a palindrome, update the answer. Since i increases,
            # the cubes are checked in increasing order. This will always
            # store the largest valid palindromic cube found so far.
            ans = cube
            
        # Proceed to the next integer.
        i += 1

    # Print the final answer to standard output.
    print(ans)

solve()