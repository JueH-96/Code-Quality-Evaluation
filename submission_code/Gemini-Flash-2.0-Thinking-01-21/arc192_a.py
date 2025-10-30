import sys

def solve():
    """
    Determines if there exists a good string S of length N such that
    a binary sequence A of length N can be transformed into all ones
    using specified operations.
    """
    # Read the input N
    N = int(sys.stdin.readline())

    # Read the input sequence A
    # Adjust A to be 0-indexed for easier modulo arithmetic
    A = list(map(int, sys.stdin.readline().split()))

    # According to the analysis, a good string exists if and only if
    # there are no three consecutive zeros in the sequence A, considering
    # the sequence cyclically.
    # An operation based on S_i S_{i+1} S_{i+2} can change A_i and A_{i+1} to 1.
    # The patterns for the operation are 'ARC' and 'CRA', both requiring 'R'
    # in the middle. This implies that for an operation to be possible at index j
    # (based on S_j S_{j+1} S_{j+2}), S_{j+1} must be 'R', and S_j, S_{j+2} must be
    # 'A' and 'C'. This structure prevents 'RR' from appearing in a good string S.
    # The set of indices in A that can be flipped by operations starting at indices
    # in a set J is F = J union (J+1) (modulo N).
    # If there are three consecutive zeros, say A_i=0, A_{i+1}=0, A_{i+2}=0
    # (indices modulo N), then we need {i, (i+1)%N, (i+2)%N} to be a subset
    # of F.
    # This requires F to contain three consecutive indices.
    # The set F = J union (J+1) contains three consecutive indices k, (k+1)%N, (k+2)%N
    # if and only if there exists j such that j is in J and (j+1)%N is in J.
    # However, the condition for j in J (S_{j+1}='R', S_j, S_{j+2} in {'A','C'}) and
    # (j+1)%N in J (S_{(j+2)%N}='R', S_{(j+1)%N}, S_{(j+3)%N} in {'A','C'})
    # simultaneously requires S_{(j+1)%N}='R' and S_{(j+2)%N}='R', which is
    # impossible in a good string.
    # Therefore, if there are three consecutive zeros in A (cyclically), no good string exists.
    # Conversely, if there are no three consecutive zeros, it is possible to
    # construct a string S (e.g., by placing 'R's appropriately) that allows
    # operations to cover all zeros. Blocks of one or two zeros can be covered.

    # Check for three consecutive zeros (cyclically)
    has_three_consecutive_zeros = False
    # Iterate through all possible starting positions i from 0 to N-1
    for i in range(N):
        # Check A_i, A_{i+1}, A_{i+2} using modulo arithmetic for cyclic wrap-around
        if A[i] == 0 and A[(i + 1) % N] == 0 and A[(i + 2) % N] == 0:
            has_three_consecutive_zeros = True
            break # Found three consecutive zeros, no need to check further

    # Output the result based on the finding
    if has_three_consecutive_zeros:
        print("No")
    else:
        print("Yes")

# Call the solve function to run the program
solve()