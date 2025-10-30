# YOUR CODE HERE
import sys

def is_palindrome(n):
    """Checks if the decimal representation of a positive integer n is a palindrome."""
    # Convert the number to its string representation
    s = str(n)
    # Check if the string is equal to its reverse
    return s == s[::-1]

def solve():
    """Reads input N, finds the maximum palindromic cube <= N, and prints it."""
    # Read the input integer N from standard input
    # The problem guarantees N is a positive integer <= 10^18.
    try:
        n = int(sys.stdin.readline())
    except ValueError:
        # Exit gracefully if input is not a valid integer
        # Although constraints guarantee valid input format.
        # Consider printing an error message or handling as per competition rules if needed.
        return # Or raise error

    # Initialize the variable to store the maximum palindromic cube found so far.
    # Since N is positive (N >= 1), the cube 1 (1^3) is always a potential candidate
    # because 1 is a palindrome and 1 <= N for all valid N.
    # Initializing to 0 ensures that if N < 1 (not possible by constraints) or if only 1 is found,
    # the logic works. If N >= 1, 1^3=1 will be considered.
    max_palindromic_cube = 0

    # Iterate through possible base integers x, starting from 1.
    # We are looking for K = x^3 such that K is a palindrome and K <= N.
    x = 1
    while True:
        # Calculate the cube of x.
        # Python's built-in integers handle arbitrary precision, so intermediate
        # calculations and the final cube value will not overflow for N up to 10^18.
        # x can go up to 10^6, x^3 up to 10^18.
        try:
            cube = x * x * x
            # Using x**3 is equivalent and potentially clearer
            # cube = x ** 3
        except OverflowError:
             # This is highly unlikely with Python's arbitrary precision integers
             # but included as a theoretical safeguard.
             # print("Overflow during cube calculation.", file=sys.stderr) # Avoid printing errors to stdout
             break


        # Check if the calculated cube exceeds the input limit N.
        if cube > n:
            # If the cube is greater than N, then this cube and all subsequent cubes
            # (from x+1, x+2, ...) will also be greater than N.
            # The last recorded max_palindromic_cube holds the maximum value found that is <= N.
            # We can stop the search.
            break

        # Check if the current cube is a palindrome.
        if is_palindrome(cube):
            # If the cube is a palindrome and it is less than or equal to N (implicit from the loop condition),
            # it is a candidate for our answer. Since we are iterating x in increasing order,
            # the cubes are also increasing. Thus, any new palindromic cube found will be
            # greater than or equal to the previous ones found. We update our maximum value.
            max_palindromic_cube = cube

        # Increment x to check the next integer's cube.
        # Check for potential issues if x becomes excessively large.
        # The maximum value x needs to reach is floor(N^(1/3)), which is at most 10^6 for N=10^18.
        # This loop limit is well within reasonable computation limits.
        # Add a safeguard against potential infinite loop if x somehow fails to increase, though simple increment avoids this.
        # A check like `if x > 2 * 10**6:` could be added for safety, but the `cube > n` check should suffice.
        x += 1

    # Print the final result, which is the largest palindromic cube found that is <= N.
    # If no palindromic cubes are found other than possibly 1, it will print the largest one found (at least 1 if N>=1).
    # If N was such that even 1^3 > N (e.g., hypothetical N=0), it would print 0. But N is positive.
    print(max_palindromic_cube)

# Execute the solve function to run the logic based on standard input/output.
solve()

# End of YOUR CODE HERE