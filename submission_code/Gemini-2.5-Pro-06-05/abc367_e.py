# YOUR CODE HERE
import sys

def main():
    """
    Main function to solve the problem.
    """
    # Read N, K from stdin
    try:
        n, k = map(int, sys.stdin.readline().split())
    except (IOError, ValueError):
        return  # End of input

    # Read X (1-based)
    x_list = list(map(int, sys.stdin.readline().split()))

    # Read the initial sequence A
    a = list(map(int, sys.stdin.readline().split()))

    # If K=0, no operations are performed. Print the original sequence and exit.
    if k == 0:
        print(*a)
        return

    # The transformation rule is B_i = A_{X_i}. In 0-based indexing, this is B[i] = A[X[i]-1].
    # Let p[i] = X[i]-1. The operation is B[i] = A[p[i]].
    # Applying this K times results in A_final[i] = A_initial[p^K(i)].
    # We need to compute p^K(i) for all i, which can be done efficiently using binary lifting.

    # Convert 1-based X to a 0-based mapping `p`.
    p = [val - 1 for val in x_list]

    # K can be up to 10^18. log2(10^18) is approx 59.79.
    # We need to compute jumps for powers of 2 up to 2^59.
    # So, we need 60 levels in our jump table (for j=0 to 59).
    LOG_K_MAX = 60

    # `jump[j][i]` will store the destination from index `i` after 2^j steps, i.e., p^(2^j)(i).
    jump = [[0] * n for _ in range(LOG_K_MAX)]

    # Base case: a jump of 2^0 = 1 step is defined by `p`.
    jump[0] = p

    # Precompute the jump table using doubling.
    # Recurrence: p^(2^j)(i) = p^(2^(j-1))(p^(2^(j-1))(i)).
    for j in range(1, LOG_K_MAX):
        for i in range(n):
            prev_jump_pos = jump[j-1][i]
            jump[j][i] = jump[j-1][prev_jump_pos]

    # Compute p^K(i) for each starting index `i`.
    # Decompose K into its binary representation: K = sum(b_j * 2^j).
    # p^K is the composition of p^(2^j) for all j where b_j=1.
    final_positions = [0] * n
    for i in range(n):
        current_pos = i
        # Iterate through the bits of K.
        for j in range(LOG_K_MAX):
            if (k >> j) & 1:
                # If the j-th bit of K is set, apply the precomputed 2^j jump.
                current_pos = jump[j][current_pos]
        final_positions[i] = current_pos

    # Construct the final sequence A'.
    # A'[i] = A_initial[p^K(i)] = A_initial[final_positions[i]].
    final_a = [0] * n
    for i in range(n):
        final_a[i] = a[final_positions[i]]

    # Print the result space-separated.
    print(*final_a)

if __name__ == "__main__":
    main()