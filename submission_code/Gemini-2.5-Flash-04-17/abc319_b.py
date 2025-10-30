import sys

def solve():
    """
    Reads N, computes the resulting string based on the rules, and prints it.
    """
    # Read input from standard input
    N = int(sys.stdin.readline())

    # Initialize the result string as a list of characters, length N+1
    # All characters are initially '-'
    result = ['-'] * (N + 1)

    # Iterate through each index i from 0 to N
    for i in range(N + 1):
        # Iterate through possible divisors j from 1 to 9
        # We iterate j in increasing order, so the first j found will be the smallest.
        for j in range(1, 10):
            # Check if j is a divisor of N
            if N % j == 0:
                # Calculate the corresponding interval N/j
                interval = N // j # Use integer division
                # Check if i is a multiple of N/j
                if i % interval == 0:
                    # If both conditions are met, this j is the smallest valid one
                    # for index i. Set the character and break the inner loop.
                    result[i] = str(j)
                    break # Found the smallest j, no need to check larger j for this i

    # Join the list of characters into a single string
    output_string = "".join(result)

    # Print the resulting string to standard output
    print(output_string)

# Execute the solve function
solve()