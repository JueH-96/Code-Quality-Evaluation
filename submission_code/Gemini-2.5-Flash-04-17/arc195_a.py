# YOUR CODE HERE
import sys

def solve():
    # Read N and M
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    M = int(line1[1])

    # Read sequences A and B
    # Use strip() to remove potential trailing newline character before splitting
    A = list(map(int, sys.stdin.readline().strip().split()))
    B = list(map(int, sys.stdin.readline().strip().split()))

    # Constraint check: 1 <= M <= N. If M > N, impossible to form B.
    # Although constraints guarantee M <= N, this check is harmless.
    if M > N:
        print("No")
        return

    # Compute first_idx: indices in A of the first occurrence of each element of B
    # such that the indices are strictly increasing.
    # first_idx[k] will store the smallest index i such that A[i] == B[k]
    # and i is greater than the index found for B[k-1] (if k > 0).
    first_idx = [-1] * M
    b_ptr = 0 # Pointer for sequence B (0 to M-1)
    for a_ptr in range(N): # Pointer for sequence A (0 to N-1)
        if b_ptr < M and A[a_ptr] == B[b_ptr]:
            first_idx[b_ptr] = a_ptr
            b_ptr += 1

    # If b_ptr did not reach M, it means we couldn't find all elements of B
    # in the required increasing order, so no subsequence exists.
    if b_ptr < M:
        print("No")
        return

    # Compute last_idx: indices in A of the last occurrence of each element of B
    # such that the indices are strictly increasing. We find this by iterating
    # backwards through A and matching elements of B from the end.
    # last_idx[k] will store the largest index i such that A[i] == B[k]
    # and i is less than the index found for B[k+1] (if k < M-1).
    last_idx = [-1] * M
    b_ptr = M - 1 # Pointer for sequence B (M-1 down to 0)
    for a_ptr in range(N - 1, -1, -1): # Pointer for sequence A (N-1 down to 0)
        if b_ptr >= 0 and A[a_ptr] == B[b_ptr]:
            last_idx[b_ptr] = a_ptr
            b_ptr -= 1

    # If we successfully computed first_idx, we know at least one subsequence exists.
    # The reverse pass (computing last_idx) is guaranteed to find the indices
    # for the last possible subsequence if at least one exists.

    # Compare first_idx and last_idx. If they are the same for all k (0 to M-1),
    # there is exactly one subsequence. If different for any k, there are at least two.
    found_diff = False
    for k in range(M):
        if first_idx[k] != last_idx[k]:
            found_diff = True
            break # Found a difference, no need to check further

    if found_diff:
        print("Yes")
    else:
        print("No")

# Execute the solve function
solve()