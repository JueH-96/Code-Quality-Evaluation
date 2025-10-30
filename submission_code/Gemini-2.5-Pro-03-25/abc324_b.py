# YOUR CODE HERE
import sys

def main():
    """
    Reads an integer N from standard input.
    Checks if N can be expressed as 2^x * 3^y for non-negative integers x and y.
    Prints "Yes" or "No" to standard output.
    """
    # Read input N from stdin
    n_str = sys.stdin.readline()
    try:
        n = int(n_str)
    except ValueError:
        # According to the problem constraints, the input will be a valid integer.
        # This error handling is included for robustness but may not be strictly necessary
        # in a typical competitive programming context where input format is guaranteed.
        # print("Invalid input: Not an integer.", file=sys.stderr)
        return

    # Constraint: N is a positive integer (1 <= N <= 10^18)
    # Check if N is positive, although constraints guarantee N >= 1.
    if n <= 0:
        # This case should not happen based on constraints.
        # If N=0, it cannot be 2^x * 3^y since 2^x * 3^y > 0.
        print("No")
        return

    # The core idea is that if N = 2^x * 3^y, its only prime factors are 2 and 3.
    # We can test this by repeatedly dividing N by 2 and then by 3.
    # If the remaining number after these divisions is 1, then N satisfies the condition.

    # Repeatedly divide N by 2 as long as it is divisible by 2.
    # This removes all factors of 2 from N.
    while n % 2 == 0:
        # Use integer division // to ensure n remains an integer.
        n //= 2

    # Repeatedly divide the possibly updated N by 3 as long as it is divisible by 3.
    # This removes all factors of 3 from N.
    while n % 3 == 0:
        n //= 3

    # After removing all factors of 2 and 3:
    # If the final value of N is 1, it means the original N was composed
    # solely of prime factors 2 and 3. This corresponds to N = 2^x * 3^y.
    if n == 1:
        print("Yes")
    else:
        # If the final value of N is greater than 1, it means the original N
        # had at least one prime factor other than 2 or 3.
        print("No")

# Standard boilerplate in Python to ensure the main function is called when the script executes.
if __name__ == "__main__":
    main()
# YOUR CODE HERE