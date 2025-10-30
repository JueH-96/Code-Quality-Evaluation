import sys

def solve():
    """
    Reads input, solves the problem, and prints the output.
    """
    try:
        # Fast I/O for reading large inputs
        line1 = sys.stdin.readline()
        if not line1:
            return
        N, K = map(int, line1.split())
        A = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        return

    # Using max(A) to size arrays is an optimization for cases where
    # the maximum value is smaller than the global constraint of 10^6.
    # N >= 1 ensures A is not empty.
    max_val = max(A)
    limit = max_val + 1

    # Step 1: Count frequency of each number in A.
    freq = [0] * limit
    for x in A:
        freq[x] += 1

    # Step 2: For each g, count how many numbers in A are its multiples.
    # count[g] = sum of frequencies of all multiples of g.
    count = [0] * limit
    for g in range(1, limit):
        for m in range(g, limit, g):
            count[g] += freq[m]

    # Step 3: Find the maximum possible GCD for each value from 1 to max_val.
    # ans_val[x] will store max({g | g divides x and count[g] >= K}).
    ans_val = [0] * limit
    # Iterate g from high to low to find the largest valid GCD first.
    for g in range(limit - 1, 0, -1):
        # If g can be a GCD for K or more elements...
        if count[g] >= K:
            # ...it's a candidate answer for all its multiples.
            for m in range(g, limit, g):
                # If the answer for m is not yet found, g is the largest such GCD.
                if ans_val[m] == 0:
                    ans_val[m] = g

    # Step 4: Print the pre-computed answer for each element in the input array A.
    for x in A:
        print(ans_val[x])

solve()