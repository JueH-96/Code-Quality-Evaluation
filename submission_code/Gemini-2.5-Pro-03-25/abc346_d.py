# YOUR CODE HERE
import sys

def solve():
    # Read input N: length of the string
    N = int(sys.stdin.readline())
    # Read input S: the binary string
    S = sys.stdin.readline().strip()
    # Read input C: list of costs for flipping character at each position
    C = list(map(int, sys.stdin.readline().split()))

    # A "good string" T is defined as a binary string of length N
    # with exactly one index i (1 <= i <= N-1) such that T[i] == T[i+1].
    # This implies T must be alternating everywhere except at position i.
    # There are two basic alternating patterns:
    # P0: starts with 0 (e.g., 01010...)
    # P1: starts with 1 (e.g., 10101...)

    # In Python's 0-based indexing:
    # P0[i] has value i % 2
    # P1[i] has value (i + 1) % 2

    # Calculate the cost array for transforming S to P0.
    # costs_P0[i] is the cost to change S[i] to match P0[i] (0 if already matches).
    costs_P0 = [0] * N
    # Calculate the cost array for transforming S to P1.
    # costs_P1[i] is the cost to change S[i] to match P1[i] (0 if already matches).
    costs_P1 = [0] * N

    for i in range(N):
        # Calculate cost for P0 pattern match at index i
        target_P0 = i % 2  # Expected character in P0
        if int(S[i]) != target_P0:
            costs_P0[i] = C[i]  # Cost is C[i] if S[i] needs flipping
        
        # Calculate cost for P1 pattern match at index i
        target_P1 = (i + 1) % 2 # Expected character in P1
        if int(S[i]) != target_P1:
            costs_P1[i] = C[i]  # Cost is C[i] if S[i] needs flipping

    # Compute prefix sums of the costs.
    # PC0[k] will store the sum of costs_P0[0...k-1]. This is the cost to make the prefix S[0..k-1] match P0.
    PC0 = [0] * (N + 1)
    for k in range(N):
        PC0[k+1] = PC0[k] + costs_P0[k]

    # PC1[k] will store the sum of costs_P1[0...k-1]. This is the cost to make the prefix S[0..k-1] match P1.
    PC1 = [0] * (N + 1)
    for k in range(N):
        PC1[k+1] = PC1[k] + costs_P1[k]

    # Compute suffix sums of the costs.
    # SC0[k] will store the sum of costs_P0[k...N-1]. This is the cost to make the suffix S[k..N-1] match P0.
    SC0 = [0] * (N + 1)
    for k in range(N - 1, -1, -1):
        SC0[k] = SC0[k+1] + costs_P0[k]

    # SC1[k] will store the sum of costs_P1[k...N-1]. This is the cost to make the suffix S[k..N-1] match P1.
    SC1 = [0] * (N + 1)
    for k in range(N - 1, -1, -1):
        SC1[k] = SC1[k+1] + costs_P1[k]

    # Initialize the minimum total cost found so far to infinity.
    min_total_cost = float('inf')

    # Iterate through all possible positions k (1-based index, 1 <= k <= N-1) 
    # where the single pair of identical adjacent characters T_k = T_{k+1} could be located.
    # In 0-based indexing, this means the characters at indices k-1 and k are identical,
    # and the split between potentially different patterns happens after index k-1.
    # The prefix part is indices 0..k-1. The suffix part is indices k..N-1.
    for k in range(1, N):
        
        # A good string T with T_k = T_{k+1} must be formed by taking a prefix from one pattern (P0 or P1)
        # and a suffix from the OTHER pattern (P1 or P0 respectively), such that the characters at the boundary match.
        # There are two possibilities for constructing such a T for a fixed k:

        # Case 1: Target string T constructed using Prefix P0 and Suffix P1.
        # The cost is the sum of costs to make S[0..k-1] match P0[0..k-1] (prefix cost)
        # plus the sum of costs to make S[k..N-1] match P1[k..N-1] (suffix cost).
        # Prefix cost = PC0[k] (sum of costs_P0 for indices 0 to k-1)
        # Suffix cost = SC1[k] (sum of costs_P1 for indices k to N-1)
        cost_A = PC0[k] + SC1[k]
        # Check if this configuration forms a valid good string T.
        # The characters at the boundary (0-based indices k-1 and k) must be equal.
        # T[k-1] would be P0[k-1] = (k-1)%2. T[k] would be P1[k] = (k+1)%2.
        # (k-1)%2 == (k+1)%2 is always true since k-1 and k+1 have the same parity.
        # So this configuration always produces a valid candidate string. Update minimum cost.
        min_total_cost = min(min_total_cost, cost_A)

        # Case 2: Target string T constructed using Prefix P1 and Suffix P0.
        # The cost is the sum of costs to make S[0..k-1] match P1[0..k-1] (prefix cost)
        # plus the sum of costs to make S[k..N-1] match P0[k..N-1] (suffix cost).
        # Prefix cost = PC1[k] (sum of costs_P1 for indices 0 to k-1)
        # Suffix cost = SC0[k] (sum of costs_P0 for indices k to N-1)
        cost_B = PC1[k] + SC0[k]
        # Check if this configuration forms a valid good string T.
        # The characters at the boundary (0-based indices k-1 and k) must be equal.
        # T[k-1] would be P1[k-1] = k%2. T[k] would be P0[k] = k%2.
        # k%2 == k%2 is always true.
        # So this configuration always produces a valid candidate string. Update minimum cost.
        min_total_cost = min(min_total_cost, cost_B)

    # Output the overall minimum cost found across all possible positions k.
    print(min_total_cost)

# Execute the solution function
solve()