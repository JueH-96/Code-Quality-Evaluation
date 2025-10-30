# YOUR CODE HERE
import sys

def solve():
    """
    Main function to solve the problem.
    """
    try:
        # Fast I/O
        # It's faster to read all input at once
        lines = sys.stdin.readlines()
        if not lines: return # Handle empty input case for local testing
        N, Q = map(int, lines[0].split())
        queries = [int(line) for line in lines[1:]]
    except (IOError, ValueError):
        # Fallback for empty input
        return

    # The problem asks for which integers K it is possible to construct an N x N
    # binary matrix that has exactly K fixed elements.

    # Two matrices A and B are "similar" if they have the same row sums and column sums.
    # An element A_ij is "fixed" if for any matrix B similar to A, B_ij = A_ij.

    # The structure of fixed elements can be analyzed using a graph representation.
    # The number of non-fixed elements for a matrix A is given by sum(r_k * c_k),
    # where the sum is over the strongly connected components (SCCs) of an associated
    # directed graph, and r_k, c_k are the number of row and column vertices in SCC k.
    # The number of fixed elements is N*N - sum(r_k * c_k).

    # The core of the problem is to find all possible values for this sum.
    # A sum is possible if there exist partitions of N, {r_k} and {c_k},
    # such that sum(r_k) = N, sum(c_k) = N, and the sum is sum(r_k * c_k).
    # This is equivalent to partitioning a set of N rows and N columns into groups,
    # where the k-th group has r_k rows and c_k columns.

    # We can use dynamic programming to find all possible values of sum(r_k * c_k).
    # Let dp[i][j] be a bitmask representing the set of all possible sums for
    # partitions of i rows and j columns.
    # A partition of (i, j) can be formed by taking a partition of a smaller
    # state (a, b) and adding a new group (part) of size (p, q),
    # where i = a+p and j = b+q. The sum increases by p*q.
    # This leads to the recurrence:
    # dp[i][j] = OR over all (0<=a<=i, 0<=b<=j, not (a=i and b=j)) of (dp[a][b] << (i-a)*(j-b))

    max_sum = N * N
    # dp[i][j] stores a bitmask of achievable sums for partitions of i and j.
    dp = [[0] * (N + 1) for _ in range(N + 1)]
    dp[0][0] = 1  # Base case: sum 0 for partitions of (0,0) is possible.

    # O(N^4) dynamic programming approach.
    # Python's arbitrary-precision integers handle the large bitmasks.
    for i in range(N + 1):
        for j in range(N + 1):
            if i == 0 and j == 0:
                continue
            
            # The partition of (i,j) is formed by taking a partition of (a,b)
            # and adding a new part (p, q), where i=a+p, j=b+q.
            # Sum for the new partition = (sum for partition of (a,b)) + p*q
            # Iterate through all possible previous states (a,b).
            current_sums = 0
            for a in range(i + 1):
                for b in range(j + 1):
                    if a == i and b == j:
                        continue
                    
                    p = i - a
                    q = j - b
                    
                    prev_sums_mask = dp[a][b]
                    if prev_sums_mask:
                        shift = p * q
                        # The maximum possible sum is N*N. A shift by p*q can be at most N*N.
                        # Total sum cannot exceed N*N. Python handles large integers,
                        # but we can prune shifts that would certainly exceed the max sum.
                        # Smallest sum in prev_sums_mask is 0 (always possible).
                        if shift > max_sum:
                            continue
                        current_sums |= (prev_sums_mask << shift)
            dp[i][j] = current_sums

    possible_non_fixed_sums = dp[N][N]
    
    # Store possible K values in a set for O(1) lookups
    possible_k_values = set()
    for s in range(max_sum + 1):
        if (possible_non_fixed_sums >> s) & 1:
            possible_k_values.add(N * N - s)
            
    # Handle the queries
    for k in queries:
        if k in possible_k_values:
            print("Yes")
        else:
            print("No")

solve()