# YOUR CODE HERE
import sys

def main():
    """
    Reads input, solves the problem, and prints the output.
    This solution runs in O(N) time complexity.
    """
    try:
        # Read N from stdin.
        line = sys.stdin.readline()
        if not line:
            return
        N = int(line)
    except (ValueError, IndexError):
        return

    # Base case: For N <= 1, the number of APs is N.
    if N <= 1:
        if N == 1:
            # Consume the line with the single number.
            sys.stdin.readline()
        print(N)
        return

    # Read the sequence A of N integers.
    try:
        A = list(map(int, sys.stdin.readline().split()))
    except (ValueError, IndexError):
        return

    # 1. Start with N, counting all subarrays of length 1.
    total_count = N

    # 2. Count APs of length >= 2.
    # We iterate through the implicit difference array D (where D_i = A[i+1]-A[i])
    # to find runs of identical differences.
    i = 0
    while i < N - 1:
        # `j` tracks the end of the current run of identical differences.
        j = i
        
        # The common difference for this potential run.
        diff = A[i + 1] - A[i]
        
        # Extend the run as long as the difference remains the same.
        # We check A[j+2] - A[j+1], so j+1 must be less than N-1.
        while j + 1 < N - 1 and A[j + 2] - A[j + 1] == diff:
            j += 1
        
        # The run of identical differences is from index `i` to `j`.
        # The length of this run is `m = j - i + 1`.
        m = j - i + 1
        
        # This run corresponds to an AP segment of length k = m + 1 in A.
        # This segment contains m * (m + 1) // 2 APs of length >= 2.
        # We add this to our total count.
        total_count += m * (m + 1) // 2
        
        # Move `i` to the start of the next potential run.
        i = j + 1
        
    print(total_count)

if __name__ == "__main__":
    main()