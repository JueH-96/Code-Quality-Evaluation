import sys

# It is good practice to read from stdin and write to stdout
# for programming contest-style problems.
def solve():
    """
    Reads input, solves the problem, and prints the result.
    """
    try:
        # Read the number of buildings, N.
        N_str = sys.stdin.readline()
        if not N_str:
            # Handle empty input
            return
        N = int(N_str)
    except (IOError, ValueError):
        # Handle cases where input is not a valid integer.
        return

    # If N is 0 or 1, the answer is N.
    if N <= 1:
        # If N=1, there's still a line of heights to read.
        if N == 1:
            sys.stdin.readline()
        print(N)
        return
    
    # Read the heights of the N buildings.
    H = list(map(int, sys.stdin.readline().split()))
    
    # The minimum answer is 1 (choosing any single building).
    max_count = 1

    # Iterate through all possible common differences `d`.
    for d in range(1, N):
        # Optimization: If the longest possible sequence for this difference `d`
        # cannot exceed the current max_count, we can stop checking `d` and all larger `d`'s.
        if (N - 1) // d + 1 <= max_count:
            break

        # For a fixed `d`, we check all `d` possible starting alignments.
        # Any arithmetic progression with difference `d` is a subsequence of one of these.
        for start_offset in range(d):
            
            # Optimization: If this specific sequence is not long enough to beat max_count, skip it.
            if (N - 1 - start_offset) // d + 1 <= max_count:
                continue

            current_run_length = 0
            # Use a sentinel value for last_height that cannot be in the input.
            # Heights are positive, so -1 is safe.
            last_height = -1 

            # Find the longest run of identical heights in this arithmetic progression of indices.
            for i in range(start_offset, N, d):
                if H[i] == last_height:
                    current_run_length += 1
                else:
                    # The run of 'last_height' is broken. Update max_count.
                    max_count = max(max_count, current_run_length)
                    # Start a new run.
                    last_height = H[i]
                    current_run_length = 1
            
            # After the loop, check the last run.
            max_count = max(max_count, current_run_length)

    print(max_count)

# Execute the solution
solve()