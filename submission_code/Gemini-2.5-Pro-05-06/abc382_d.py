import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())

    # Calculate M_prime as described in the logic.
    # M_prime = M - (N-1)*10 means we are looking for sequences B such that
    # 1 <= B_1 <= B_2 <= ... <= B_N <= M_prime.
    M_prime = M - (N - 1) * 10

    results = []  # To store the generated A sequences
    current_B_path = []  # To store the current B sequence being built

    def generate_B_sequences_recursive(k, min_val_for_current_B_k):
        # k: current index in B to fill (1-indexed, from 1 to N)
        # min_val_for_current_B_k: The value B_k must be at least this.
        # (This is B_{k-1} if k > 1, or 1 if k = 1, based on initial call)
        
        if k == N + 1:
            # A complete B sequence of length N is in current_B_path.
            # Convert it to an A sequence.
            A_sequence = []
            for i in range(N): # i is 0-indexed for path access (B_1 is path[0], etc.)
                # A_{i+1} = B_{i+1} + i*10 where B_{i+1} is current_B_path[i]
                A_val = current_B_path[i] + i * 10
                A_sequence.append(A_val)
            results.append(A_sequence)
            return

        # Iterate through possible values for B_k.
        # B_k can range from min_val_for_current_B_k up to M_prime.
        # (This is because B_k <= B_{k+1} <= ... <= B_N <= M_prime, so B_k itself must be <= M_prime)
        for val_Bk in range(min_val_for_current_B_k, M_prime + 1):
            current_B_path.append(val_Bk)
            # For B_{k+1}, its minimum value must be the chosen val_Bk (due to B_k <= B_{k+1})
            generate_B_sequences_recursive(k + 1, val_Bk)
            current_B_path.pop() # Backtrack: remove val_Bk to try other options

    # Initial call to the recursive function:
    # Start filling B_1 (k=1). Its minimum value is 1 (as B_1 >= 1).
    generate_B_sequences_recursive(1, 1)

    # Output results
    sys.stdout.write(str(len(results)) + "
")
    for seq in results:
        # Each element in seq is an integer. Convert to string for join.
        sys.stdout.write(" ".join(map(str, seq)) + "
")

if __name__ == '__main__':
    solve()