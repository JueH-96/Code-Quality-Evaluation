# YOUR CODE HERE
import sys

def solve():
    """
    Reads an integer N and generates a string based on the specified rules.
    """
    # Read the integer N from standard input.
    try:
        N = int(sys.stdin.readline())
    except (IOError, ValueError):
        # This handles potential read errors, though constraints suggest valid input.
        return

    # Use a list to build the output string efficiently.
    result_chars = []

    # Iterate through each index i from 0 to N.
    for i in range(N + 1):
        # Default character for s_i is '-', used if no suitable j is found.
        char_for_i = '-'

        # Iterate through possible divisors j from 1 to 9.
        # Looping from 1 to 9 ensures that the first j found is the smallest.
        for j in range(1, 10):
            # Condition 1: j must be a divisor of N.
            if N % j == 0:
                # Calculate d = N/j. Integer division is safe as N % j == 0.
                d = N // j

                # Condition 2: i must be a multiple of d.
                if i % d == 0:
                    # If both conditions are met, we've found the smallest valid j.
                    # Set the character for this index to the digit of j.
                    char_for_i = str(j)
                    # Break the inner loop since we've found the answer for this i.
                    break
        
        # Append the determined character to our list of characters.
        result_chars.append(char_for_i)

    # Join the list of characters into a single string and print to standard output.
    print("".join(result_chars))

solve()