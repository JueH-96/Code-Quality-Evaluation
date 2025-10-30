import sys

def solve():
    """
    Calculates the number of distinct possible XOR sums of stone counts
    after performing the specified operation any number of times.
    """
    # Read input
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Set to store distinct possible values of (Sum(S) XOR Xor(S))
    # The set of possible final XOR sums B_1^...^B_N is { X_total ^ (Sum(S) XOR Xor(S)) | S is a subset }
    # The number of distinct values of X_total ^ Y_S is the same as the number of distinct values of Y_S
    # where Y_S = Sum(S) XOR Xor(S).
    possible_Y_values = set()

    # Iterate through all 2^N possible subsets of bags
    # Each mask from 0 to 2^N - 1 represents a unique subset.
    # If the i-th bit is set in the mask, the i-th bag is included in the subset S.
    for mask in range(1 << N):
        current_sum_S = 0
        current_xor_S = 0

        # Iterate through each bag to check if it's in the current subset S
        for i in range(N):
            if (mask >> i) & 1: # Check if the i-th bit is set
                # If bag i is in subset S
                current_sum_S += A[i]
                current_xor_S ^= A[i]

        # Calculate Y_S = (Sum of elements in S) XOR (XOR sum of elements in S)
        # According to the derivation, the set of reachable XOR sums is
        # { (XOR sum of elements NOT in S) XOR (Sum of elements in S) | S is a subset }
        # Let X_total = XOR sum of all A_i.
        # XOR sum of elements NOT in S = X_total XOR (XOR sum of elements in S)
        # So, reachable XOR sum = (X_total XOR Xor(S)) XOR Sum(S)
        # = X_total XOR (Xor(S) XOR Sum(S))
        # Let Y_S = Sum(S) XOR Xor(S).
        # The set of reachable XOR sums is { X_total XOR Y_S | S is a subset }.
        # The number of distinct values in this set is the same as the number of distinct values of Y_S.
        Y_S = current_sum_S ^ current_xor_S
        possible_Y_values.add(Y_S)

    # The answer is the number of distinct values of Y_S found
    print(len(possible_Y_values))

# Execute the solve function
solve()