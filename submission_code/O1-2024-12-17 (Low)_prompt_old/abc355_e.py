def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, L, R = map(int, input_data)

    # Explanation of the approach:
    # We will determine sum(A_L + A_{L+1} + ... + A_R) mod 100
    # by computing prefix sums mod 100:
    #     prefix(X) = sum(A_0 + A_1 + ... + A_X) mod 100
    # and using:
    #     sum(L..R) mod 100 = [prefix(R) - prefix(L - 1)] mod 100
    #
    # To get prefix(X) mod 100, we decompose [0..X] into disjoint
    # blocks of length 2^i, all of which start at consecutive indices.
    # For each block, we make one query "? i j" to get the sum over
    # that block mod 100. Adding them up (mod 100) gives prefix(X).
    #
    # The total number of questions for prefix(R) and prefix(L-1) is
    # at most 2 * N (because N <= 18), which does not exceed the
    # minimal required number m for this problem.

    import math
    sys.stdout.flush()

    # Interactive query function.
    # In a real interactive environment, we would:
    #   1) print("? i j")
    #   2) flush output
    #   3) read the judge's response
    # For this skeleton code, we will just read from standard input
    # to mimic the interaction. Of course, this won't run properly
    # here without a judge providing the responses. This is the shape
    # of the solution.
    #
    # We'll name it query(i, j). After printing "? i j", we read T.
    # If T == -1, we must terminate (judge indicates error or limit).
    def query(i, j):
        # Ask the question
        print(f"? {i} {j}", flush=True)
        # Read the response
        T = int(sys.stdin.readline())
        if T == -1:
            # Judge indicates error => must terminate immediately
            sys.exit(0)
        return T

    # Compute prefix sum A_0..A_X mod 100 by splitting [0..X] into
    # intervals of length 2^i that start at consecutive indices.
    def prefix_sum_mod100(X):
        if X < 0:
            return 0

        total = 0
        start = 0
        length = X + 1
        while length > 0:
            i = length.bit_length() - 1  # largest i so that 2^i <= length
            block_size = 1 << i         # 2^i
            j = start >> i              # because l = 2^i * j = start
            # ask the sum for A_start..A_{start + block_size - 1}
            val = query(i, j)
            total = (total + val) % 100
            start += block_size
            length -= block_size

        return total

    # Calculate the desired sum (L..R) mod 100 = prefix(R) - prefix(L-1) mod 100
    sum_LR = prefix_sum_mod100(R) - prefix_sum_mod100(L - 1)
    sum_LR %= 100

    # Print the final answer and terminate
    print(f"! {sum_LR}")
    sys.stdout.flush()

# Normally, you would call solve() here in an interactive environment.
# We include the call as requested:
solve()