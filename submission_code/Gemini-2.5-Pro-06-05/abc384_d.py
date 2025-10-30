# YOUR CODE HERE
import sys
from bisect import bisect_right

def main():
    """
    This function reads input, solves the problem, and prints the output.
    """
    
    # Read input from stdin
    try:
        input_stream = sys.stdin
        line1 = input_stream.readline()
        if not line1: return
        N_str, S_str = line1.split()
        N = int(N_str)
        S = int(S_str)
        line2 = input_stream.readline()
        if not line2: return
        A = list(map(int, line2.split()))
    except (IOError, ValueError):
        # Handle potential empty lines or invalid format
        return

    # --- Core Algorithm ---
    
    SUM_N = sum(A)

    # To handle all cyclic subsequences of length up to N, we use a doubled array.
    B = A + A
    P = [0] * (2 * N + 1)
    for i in range(2 * N):
        P[i + 1] = P[i] + B[i]

    # Group prefix sums P[k] by their remainder modulo SUM_N.
    # M maps: remainder -> list of (index k, prefix_sum P[k])
    M = {}
    for k in range(2 * N + 1):
        rem = P[k] % SUM_N
        if rem not in M:
            M[rem] = []
        M[rem].append((k, P[k]))

    # For efficient binary search, create a map from remainder to a sorted list of indices.
    M_keys = {rem: [item[0] for item in items] for rem, items in M.items()}

    # Iterate through all cyclically distinct start points, which are indices 0 to N-1.
    # P[i] is the prefix sum before the start of the subsequence.
    for i in range(N):
        Pi = P[i]
        
        # We need to find an end point k such that:
        # P[k] % SUM_N == (S + Pi) % SUM_N
        target_rem = (S + Pi) % SUM_N

        if target_rem in M:
            candidates = M[target_rem]
            cand_keys = M_keys[target_rem]

            # Find the first candidate index k that is strictly greater than i.
            # A subsequence must be non-empty, so its length k-i must be at least 1.
            idx = bisect_right(cand_keys, i)

            if idx < len(cand_keys):
                k, Pk = candidates[idx]

                # The length of the base subsequence part is k - i.
                # We are considering base parts of length up to N.
                if k <= i + N:
                    sub_sum = Pk - Pi
                    
                    # We need S >= sub_sum. The modulo condition is guaranteed by our
                    # choice of k. If S >= sub_sum, it means S can be formed by
                    # sub_sum plus some non-negative number of full periods SUM_N.
                    if sub_sum <= S:
                        print("Yes")
                        return

    # If no such subsequence is found after checking all possibilities.
    print("No")

if __name__ == "__main__":
    main()