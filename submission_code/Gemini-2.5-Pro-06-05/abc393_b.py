# YOUR CODE HERE
import sys

def solve():
    """
    This function reads a string from standard input, finds the number of
    'ABC' sequences at even intervals, and prints the result to standard output.
    """
    # Read the input string from stdin
    S = sys.stdin.readline().strip()
    N = len(S)

    # Initialize a counter for the number of valid triples
    count = 0

    # Iterate through all possible pairs of indices (i, j) where i < j.
    # This O(N^2) approach is efficient enough for N <= 100.
    for i in range(N):
        for j in range(i + 1, N):
            # We are looking for a triple of indices (i, j, k) that satisfy:
            # 1. 0 <= i < j < k < N
            # 2. j - i = k - j  (even intervals)
            # 3. S[i] == 'A', S[j] == 'B', S[k] == 'C'

            # From condition (2), we can determine the index k from i and j:
            # k = j + (j - i) = 2*j - i
            k = 2 * j - i

            # Now, we check the other conditions.
            # The loops ensure i < j.
            # The calculation k = j + (j - i) ensures k > j because j - i > 0.
            # So, we only need to check if k is a valid index (within the string bounds).
            if k < N:
                # If k is a valid index, check if the characters match the required sequence.
                if S[i] == 'A' and S[j] == 'B' and S[k] == 'C':
                    # If all conditions are met, we have found a valid triple.
                    count += 1

    # Print the final count to stdout.
    print(count)

# Execute the solution
solve()