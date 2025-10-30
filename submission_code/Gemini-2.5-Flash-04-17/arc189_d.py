import sys

# Function to compute L[k]: the leftmost index (0-based) included in the final blob for starting slime k
# L[k] = i + 1, where i is the largest index < k such that A[i] >= sum(A[i+1...k])
def compute_L(N, A):
    L = [0] * N
    stack = [] # Stores indices i
    # The stack stores indices i from left to right (increasing).
    # current_sum_from_k accumulates A[k] and then A[i] from popped indices.
    # Due to the stack's popping order, when considering A[i], current_sum_from_k
    # is precisely the sum of elements A[p] for indices p that were in the stack
    # on top of i and subsequently popped, plus A[k]. This sum is exactly sum(A[i+1...k]).
    for k in range(N):
        current_sum_from_k = A[k] # Initial sum: A[k]
        while stack:
            i = stack.pop()
            # current_sum_from_k now holds sum(A[i_next + 1 ... k]), where i_next is the index just above i that was popped.
            # If i was the top, current_sum_from_k is just A[k].
            # The sum required is sum(A[i+1 ... k]), which is current_sum_from_k in this stack implementation.
            if A[i] >= current_sum_from_k:
                L[k] = i + 1
                stack.append(i) # i is the barrier, push it back
                break
            else:
                current_sum_from_k += A[i] # i is not a barrier, add its value to the sum for the next check
        if not stack:
            L[k] = 0
        stack.append(k) # Push k for future k' > k

    return L

# Function to compute R[k]: the rightmost index (0-based) included in the final blob for starting slime k
# R[k] = j - 1, where j is the smallest index > k such that A[j] >= sum(A[k...j-1])
def compute_R(N, A):
    R = [0] * N
    stack = [] # Stores indices j
    # Process from right to left. Stack stores indices j from right to left (decreasing).
    # current_sum_from_k accumulates A[k] and then A[j] from popped indices.
    # Due to the stack's popping order, when considering A[j], current_sum_from_k
    # is precisely the sum of elements A[p] for indices p that were in the stack
    # on top of j and subsequently popped, plus A[k]. This sum is exactly sum(A[k...j-1]).
    for k in range(N - 1, -1, -1):
        current_sum_from_k = A[k] # Initial sum: A[k]
        while stack:
            j = stack.pop()
            # current_sum_from_k now holds sum(A[k ... j_prev - 1]), where j_prev is the index just below j that was popped.
            # If j was the top, current_sum_from_k is just A[k].
            # The sum required is sum(A[k ... j-1]), which is current_sum_from_k in this specific stack implementation.
            if A[j] >= current_sum_from_k:
                R[k] = j - 1
                stack.append(j) # j is the barrier
                break
            else:
                current_sum_from_k += A[j] # j is not a barrier, add its value to the sum for the next check
        if not stack:
            R[k] = N - 1
        stack.append(k) # Push k for future k' < k

    return R

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Calculate prefix sums for efficient range sum queries
    P = [0] * (N + 1)
    for i in range(N):
        P[i+1] = P[i] + A[i]

    L = compute_L(N, A)
    R = compute_R(N, A)

    # Calculate the final size for each starting slime k
    results = []
    for k in range(N):
        # The final blob for starting slime k covers original indices from L[k] to R[k]
        final_size = P[R[k] + 1] - P[L[k]]
        results.append(final_size)

    print(*results)

solve()