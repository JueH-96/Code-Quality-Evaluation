import sys

# Function to perform ceiling division
# a: numerator, b: denominator (b > 0)
def ceil_div(a, b):
    if b <= 0:
        # Should not happen based on constraints A_i, B_j >= 1
        raise ValueError("Denominator must be positive")
    if a >= 0:
        return (a + b - 1) // b
    else:
        # Python's // operator rounds down (towards negative infinity)
        # If a is a multiple of b, a // b is the correct integer result, which is also the ceiling.
        # If a is not a multiple of b, a // b is floor, ceiling is floor + 1.
        if a % b == 0:
            return a // b
        else:
            return a // b + 1

# Function to count the number of triples (i, j, k) such that
# A[i]*B[j] + B[j]*C[k] + C[k]*A[i] >= X
# This function runs in O(N^2) time after sorting A, B, C.
def count_greater_equal(X, N, A, B, C):
    count = 0
    
    # The expression is A[i]*B[j] + B[j]*C[k] + C[k]*A[i] >= X
    # We can rearrange to count k for fixed i and j:
    # C[k]*(A[i] + B[j]) >= X - A[i]*B[j]
    # Since A[i] >= 1 and B[j] >= 1, A[i] + B[j] >= 2 > 0.
    # C[k] >= (X - A[i]*B[j]) / (A[i] + B[j])
    # Let V_ij = ceil( (X - A[i]*B[j]) / (A[i] + B[j]) )
    # We need to count k such that C[k] >= V_ij

    # The total count is sum over i, j of (number of k such that C[k] >= V_ij)
    # Number of k such that C[k] >= V is N - bisect_left(C, V) if C is sorted ascending.

    # We sum over i, and for each i, sum over j using a two-pointer approach
    # to efficiently calculate sum(N - bisect_left(C, V_ij)) over j.
    # For a fixed i, V_ij = ceil_div(X - A[i]*B[j], A[i]+B[j]).
    # As j increases, B[j] increases, A[i]+B[j] increases, A[i]*B[j] increases.
    # The numerator (X - A[i]*B[j]) decreases. The denominator (A[i]+B[j]) increases.
    # The value (X - A[i]*B[j]) / (A[i]+B[j]) decreases as j increases.
    # V_ij = ceil_div(...) also decreases or stays the same as j increases.
    # Let p_j = bisect_left(C, V_ij). p_j decreases or stays the same as j increases.
    # We need to sum (N - p_j) over j for fixed i.
    # N - p_j increases or stays the same as j increases.

    # Two-pointer approach for the inner sum over j (for fixed i):
    # We want to calculate sum_{j=0}^{N-1} (N - bisect_left(C, V_ij))
    # Initialize a pointer `ptr_c` for array C.
    # As j increases, V_ij decreases. The index `bisect_left(C, V_ij)` decreases.
    # The count `N - bisect_left(C, V_ij)` increases.

    for i in range(N):
        inner_count_for_i = 0
        # `ptr_c` will point to the index *before* the first element >= V_ij.
        # So, elements from `ptr_c + 1` to `N-1` are >= V_ij. Number is (N-1) - (ptr_c + 1) + 1 = N - 1 - ptr_c.
        # Initialize ptr_c to N-1 (last index).
        ptr_c = N - 1 
        
        for j in range(N):
            num = X - A[i] * B[j]
            den = A[i] + B[j]
            V_ij = ceil_div(num, den)
            
            # Move ptr_c left while C[ptr_c] >= V_ij
            # After the loop, ptr_c will be the index of the last element < V_ij,
            # or -1 if all elements are >= V_ij.
            while ptr_c >= 0 and C[ptr_c] >= V_ij:
                ptr_c -= 1
            
            # The number of elements in C that are >= V_ij is N - (ptr_c + 1)
            # i.e., elements from index ptr_c + 1 to N-1.
            count_k = N - (ptr_c + 1)
            
            inner_count_for_i += count_k
        count += inner_count_for_i

    return count


# Read input
N, K = map(int, sys.stdin.readline().split())
A = list(map(int, sys.stdin.readline().split()))
B = list(map(int, sys.stdin.readline().split()))
C = list(map(int, sys.stdin.readline().split()))

# Sort arrays in ascending order
A.sort()
B.sort()
C.sort()

# Binary search for the K-th largest value.
# We are looking for the smallest value X such that the number of combinations
# (i, j, k) resulting in a value >= X is at least K.

# The possible range of values is roughly [0, 3 * 10^18].
# A[i], B[j], C[k] are up to 10^9. Max value approx 3 * (10^9)^2 = 3 * 10^18.
L = 0
R = 3 * 10**18 + 1 # Use R as exclusive upper bound

ans = R # Initialize answer outside the possible range

while L < R:
    mid = (L + R) // 2
    
    # Count how many triples (i, j, k) result in a value >= mid
    count = count_greater_equal(mid, N, A, B, C)
    
    if count >= K:
        # If count is >= K, it means there are at least K values >= mid.
        # The K-th largest value is therefore >= mid.
        # We found a value 'mid' which is a possible K-th largest or smaller.
        # The answer could be mid, or something smaller.
        # We store mid as a potential answer and try the range [L, mid].
        ans = mid
        R = mid
    else:
        # If count < K, it means there are fewer than K values >= mid.
        # The K-th largest value must be strictly less than mid.
        # We need to search in the range [mid + 1, R].
        L = mid + 1

# After the loop, L is the smallest value X such that count_greater_equal(X) >= K.
# This L is the K-th largest value.
print(L)