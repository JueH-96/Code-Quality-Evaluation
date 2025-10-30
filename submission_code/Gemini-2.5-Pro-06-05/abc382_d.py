import sys

def solve():
    """
    Reads input, solves the problem by generating sequences, and prints the output.
    """
    try:
        # Read N and M from a single line of standard input.
        line = sys.stdin.readline()
        if not line:
            return
        N, M = map(int, line.split())
    except (IOError, ValueError):
        return

    # The problem is transformed into finding non-decreasing sequences `d`
    # of length N, where 0 <= d_i <= K.
    # K is derived from the constraint A_N <= M.
    K = M - (10 * N - 9)
    
    results = []

    def generate_d_sequences(d_seq):
        """
        Recursively generates non-decreasing sequences `d` of length N.
        """
        # Base case: A complete d-sequence of length N has been formed.
        if len(d_seq) == N:
            # Transform the d-sequence back to an A-sequence.
            # Using 0-based index `i` for Python lists: A_{i+1} = 1 + 10*i + d_seq[i]
            a_seq = [1 + 10 * i + d_seq[i] for i in range(N)]
            results.append(a_seq)
            return

        # Determine the starting value for the next element in the d-sequence.
        # It must be at least the value of the previous element.
        start_val = d_seq[-1] if d_seq else 0
        
        # Iterate through possible values for the next element up to K.
        # This process naturally generates sequences in lexicographical order.
        for val in range(start_val, K + 1):
            d_seq.append(val)
            generate_d_sequences(d_seq)
            d_seq.pop()  # Backtrack to explore other possibilities.

    # Start the recursive generation with an empty sequence.
    generate_d_sequences([])

    # Print the total number of sequences found.
    print(len(results))
    
    # Print each found sequence on a new line, with elements separated by spaces.
    for seq in results:
        print(*seq)

# Execute the solution.
solve()