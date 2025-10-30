# YOUR CODE HERE
import sys

def solve():
    """
    Reads a string S from standard input and calculates the number of triples (i, j, k)
    such that 1 <= i < j < k <= |S|, j - i = k - j, S[i-1] = 'A', S[j-1] = 'B', and S[k-1] = 'C'.
    Prints the count to standard output.
    """
    # Read the input string from standard input
    s = sys.stdin.readline().strip()
    # Get the length of the string
    n = len(s)
    # Initialize the counter for valid triples
    count = 0

    # The problem asks for triples of 1-based indices (i, j, k).
    # We will use 0-based indices (p, q, r) where p = i-1, q = j-1, r = k-1.
    # The conditions become:
    # 1. 0 <= p < q < r <= n-1
    # 2. (q+1) - (p+1) = (r+1) - (q+1)  => q - p = r - q => r = 2*q - p
    # 3. s[p] == 'A'
    # 4. s[q] == 'B'
    # 5. s[r] == 'C'

    # Iterate through the possible first index p (0-based).
    # The maximum value for p can be n-3, because we need at least two subsequent indices
    # q = p+1 and r = p+2 for the smallest possible valid triple (smallest difference d=1).
    # The loop range(n - 2) goes from 0 to n-3, inclusive.
    for p in range(n - 2):
        # Iterate through the possible second index q (0-based).
        # q must be strictly greater than p (q >= p+1).
        # The maximum value for q can be n-2. This is because the third index r must satisfy r < n.
        # Since r = q + (q-p) and q-p >= 1, we have r >= q+1.
        # If q = n-1, then r >= n, which is out of bounds.
        # So, the maximum value for q is n-2.
        # The loop range(p + 1, n - 1) goes from p+1 to n-2, inclusive.
        for q in range(p + 1, n - 1):
            # Calculate the potential third index r based on the equal interval condition.
            r = 2 * q - p

            # Check if the calculated index r is valid (within the string bounds [0, n-1]).
            # The conditions p < q < r must hold.
            # p < q is guaranteed by the inner loop starting range (p + 1).
            # q < r is guaranteed because r = q + (q - p) and q - p >= 1 (since q > p).
            # So, we only need to explicitly check if r is less than n (i.e., r <= n-1).
            if r < n:
                # Check if the characters at the 0-based indices p, q, r match the required pattern 'A', 'B', 'C'
                if s[p] == 'A' and s[q] == 'B' and s[r] == 'C':
                    # If all conditions are met, increment the counter
                    count += 1

    # Print the final count to standard output
    print(count)

# Call the solve function to execute the logic when the script is run
solve()
# END OF YOUR CODE HERE