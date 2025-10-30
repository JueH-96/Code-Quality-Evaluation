import sys

def solve():
    """
    Solves the problem of finding the minimum possible value of max(B) - min(B),
    where B is a sequence formed by removing exactly K elements from A
    and concatenating the remaining N-K elements in their original order.
    """
    # Read N and K from the first line of input
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    K = int(line1[1])

    # Read the sequence A from the second line of input
    A = list(map(int, sys.stdin.readline().split()))

    # Sort the array A in non-decreasing order
    # Sorting is crucial. The minimum difference between the maximum and minimum
    # values of a sub-multiset of size N-K from A will occur when the elements
    # are chosen such that they are consecutive in the sorted version of A.
    A.sort()

    # The number of elements remaining after removing K elements
    # This is the required length of the sequence B.
    M = N - K

    # The problem asks for the minimum possible value of max(B) - min(B).
    # If we form a sequence B by keeping M elements from A, the set of values
    # in B is a sub-multiset of size M of the values in A.
    # Let the sorted values of the M elements in B be b_1 <= b_2 <= ... <= b_M.
    # We want to minimize b_M - b_1.
    # The multiset {b_1, ..., b_M} must be a sub-multiset of size M of the
    # original multiset A.
    # Consider any sub-multiset of size M from the sorted array A.
    # Let this sub-multiset be {A[j_1], A[j_2], ..., A[j_M]} where
    # 0 <= j_1 <= j_2 <= ... <= j_M < N. The range is A[j_M] - A[j_1].
    # To minimize this range, the indices j_1, ..., j_M should be as close
    # as possible. The minimum possible value of A[j_M] - A[j_1] for a
    # sub-multiset of size M is achieved when the indices are consecutive.
    # That is, when the sub-multiset is {A[i], A[i+1], ..., A[i+M-1]}
    # for some starting index i in the sorted array.
    # The range for this sub-multiset is A[i+M-1] - A[i].

    # If we select a set of M values from A that are consecutive in the sorted
    # array A, say {A[i], ..., A[i+M-1]}, the range is A[i+M-1] - A[i].
    # Can we always form a sequence B of length M from the original A
    # such that all elements in B have values within the range [A[i], A[i+M-1]]?
    # Yes. Consider the set of indices in the original array A where the values
    # are within [A[i], A[i+M-1]]. The number of such elements in A is the same
    # as the number of elements in the sorted array A within this range,
    # which is at least M (since A[i], ..., A[i+M-1] are all within this range).
    # Let these indices in the original array be p_1 < p_2 < ... < p_q, where q >= M.
    # We can select the first M indices from this list: i_1=p_1, ..., i_M=p_M.
    # The sequence B = (A[p_1], ..., A[p_M]) is a valid sequence of length M,
    # formed by removing N - M = K elements. All elements A[p_j] are within
    # the range [A[i], A[i+M-1]].
    # Therefore, min(B) >= A[i] and max(B) <= A[i+M-1], so max(B) - min(B) <= A[i+M-1] - A[i].

    # The minimum possible value of max(B) - min(B) is thus the minimum
    # difference among all contiguous subarrays of length M in the sorted array A.

    min_diff = float('inf')

    # Iterate through all possible starting indices 'i' for a contiguous
    # subarray of length M in the sorted array A.
    # The subarray is A[i : i + M]. The ending index is i + M - 1.
    # The starting index i can range from 0 up to N - M.
    # N - M is equal to K. So i ranges from 0 to K (inclusive).
    # The loop runs K + 1 times.
    for i in range(K + 1):
        # Calculate the difference between the maximum and minimum
        # elements in the current window A[i : i + M]
        diff = A[i + M - 1] - A[i]

        # Update the minimum difference found so far
        min_diff = min(min_diff, diff)

    # Print the final minimum difference
    print(min_diff)

# Call the solve function to run the program
solve()