import heapq
import math

def solve():
    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    C = list(map(int, input().split()))

    # Sort A, B, C in descending order to ensure the largest values
    # appear at the beginning of the sorted lists.
    A.sort(reverse=True)
    B.sort(reverse=True)
    C.sort(reverse=True)

    # The value for a combination (i, j, k) is A[i]*B[j] + B[j]*C[k] + C[k]*A[i].
    # We can rewrite this expression in different ways, e.g.,
    # (A[i] + B[j]) * C[k] + A[i] * B[j].
    # For fixed indices i and j, this gives a sequence of N values by varying k.
    # Since C is sorted in descending order and (A[i] + B[j]) >= 2 (as A, B >= 1),
    # the values (A[i] + B[j]) * C[k] + A[i] * B[j] for k = 0, 1, ..., N-1
    # form a sequence sorted in descending order.
    # There are N*N = N^2 such sequences, indexed by pairs (i, j).

    # We need to find the K-th largest value among all N^3 values.
    # This is equivalent to finding the K-th largest element in the union of N^2 sorted lists.
    # Since K is relatively small compared to N^3 (K <= 5e5), we can use a min-priority queue.

    # We initialize the PQ with the largest element (corresponding to k=0)
    # from a subset of the N^2 lists. The lists corresponding to smaller
    # indices i and j (which hold larger values in A and B) are more likely
    # to contain the overall largest values.
    # We heuristicly limit the initial lists to those where i < M and j < M
    # for some bound M related to K.

    # Determine the bound M for indices i and j.
    # If we consider lists (i, j) where i < M and j < M, there are M*M lists.
    # The PQ will store at most M*M elements during initialization and potentially grow
    # up to M*M + K elements in total (though typically much less if K is small).
    # We choose M such that M*M is roughly around K. A common choice is M slightly
    # larger than sqrt(K).
    M_bound = min(N, int(math.sqrt(K)) + 2)

    # Priority queue stores tuples: (-value, i, j, k)
    # We store the negative of the value to use a min-priority queue as a max-priority queue.
    # The tuple also stores the indices (i, j, k) to determine the next element in the sequence.
    pq = []

    # Initialize the priority queue with the first element (k=0) from lists (i, j)
    # where i is from 0 to min(N, M_bound)-1 and j is from 0 to min(N, M_bound)-1.
    for i in range(min(N, M_bound)):
        for j in range(min(N, M_bound)):
            # Calculate value for indices (i, j, k=0)
            value = A[i] * B[j] + B[j] * C[0] + C[0] * A[i]
            # Push (-value, i, j, k=0) into the priority queue.
            # The value can be large, Python handles arbitrary size integers.
            heapq.heappush(pq, (-value, i, j, 0))

    # Extract the K-th largest value by repeatedly popping from the PQ
    result = None
    for _ in range(K):
        # Get the largest value currently in the PQ (smallest negative value)
        neg_value, i, j, k = heapq.heappop(pq)
        result = -neg_value # The actual value

        # If the list indexed by (i, j) has more elements (i.e., k+1 < N),
        # calculate the next element (k+1) and push it into the PQ.
        if k + 1 < N:
            next_value = A[i] * B[j] + B[j] * C[k + 1] + C[k + 1] * A[i]
            # Push (-next_value, i, j, k+1) into the priority queue
            heapq.heappush(pq, (-next_value, i, j, k + 1))

    # After K extractions, the variable 'result' holds the K-th largest value.
    print(result)

solve()