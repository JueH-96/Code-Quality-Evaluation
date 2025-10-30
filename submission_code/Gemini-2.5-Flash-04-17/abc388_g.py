import sys
import bisect

def solve():
    # Read N
    N = int(sys.stdin.readline())
    # Read the array A
    A = list(map(int, sys.stdin.readline().split()))
    # Read Q
    Q = int(sys.stdin.readline())

    # Process each query
    results = []
    for _ in range(Q):
        # Read L and R (1-based indices)
        L, R = map(int, sys.stdin.readline().split())
        
        # Convert L and R to 0-based indices for Python list
        l_idx = L - 1
        r_idx = R - 1

        # The problem asks for the maximum number of kagamimochi using only
        # the mochi from index L to R (inclusive, 1-based).
        # This corresponds to the sublist A[l_idx : r_idx + 1] in 0-based indexing.
        # The mochi sizes in this sublist are A[l_idx], A[l_idx+1], ..., A[r_idx].
        # Let this sublist be S. S[k] = A[l_idx + k].
        # We want to find the maximum number of pairs (a, b) from S such that a <= b/2.
        # Since S is sorted (as A is sorted), a greedy strategy applies:
        # Always try to pair the smallest available mochi with the smallest available
        # mochi that is at least twice its size.
        # This can be efficiently implemented using two pointers on the relevant part
        # of the original array A, from l_idx to r_idx.

        count = 0
        
        # p1 is the index of the potential top mochi (smallest available)
        # p2 is the index of the potential bottom mochi for A[p1]
        p1 = l_idx
        p2 = l_idx + 1 # A bottom mochi must have a strictly larger size/index than the top mochi

        # The pointers traverse the original array within the query range [l_idx, r_idx]
        # p1 represents the index in A of the current smallest available mochi to be used as top
        # p2 represents the index in A of the current smallest available mochi to be considered as bottom for A[p1]

        while p1 <= r_idx and p2 <= r_idx:
            # Condition: A[p1] <= A[p2] / 2, which is equivalent to 2 * A[p1] <= A[p2]
            if 2 * A[p1] <= A[p2]:
                # Found a valid pair (A[p1], A[p2])
                count += 1
                
                # A[p1] is used as a top. The next smallest available mochi to be used as top
                # will be the one at index p1 + 1 (in the original array A).
                p1 += 1
                
                # A[p2] is used as a bottom. Any future bottom mochi must have an index
                # strictly greater than p2. The next candidate for bottom search starts from p2 + 1.
                p2 += 1
            else:
                # A[p2] is too small to be the bottom for A[p1].
                # A[p1] needs a larger mochi for the bottom. Since the array is sorted,
                # the next candidate for bottom is A[p2 + 1]. Keep A[p1] as the potential top.
                p2 += 1

        results.append(count)

    # Print the results for all queries
    for result in results:
        print(result)

solve()