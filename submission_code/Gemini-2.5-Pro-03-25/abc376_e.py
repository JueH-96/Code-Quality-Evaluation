# YOUR CODE HERE
import heapq
import sys

def solve():
    """
    Solves a single test case.
    Reads N, K, sequences A and B.
    Finds the minimum possible value of (max A_i in S) * (sum B_i in S)
    over all subsets S of {1..N} of size K.
    Prints the result to stdout.
    """
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    # Combine A and B into pairs (A_i, B_i)
    pairs = []
    for i in range(N):
        pairs.append((A[i], B[i]))

    # Sort the pairs based on the A_i values in non-decreasing order.
    # This allows us to process elements considering increasing maximum A_i constraints.
    pairs.sort()

    # We use a min-heap `SmallK` to store the negative values of the K smallest B_i encountered so far.
    # Storing negative values allows us to use the min-heap as a max-heap for the actual B_i values.
    # The root of the heap will contain the negative of the largest B_i among the K smallest ones.
    SmallK = []  # Stores -B_i values
    
    # `current_sum_B` tracks the sum of the B_i values currently in the conceptual set of K smallest B's.
    current_sum_B = 0
    
    # `min_product` stores the minimum product found so far, initialized to infinity.
    min_product = float('inf')

    # Iterate through the sorted pairs. Each element `pairs[k]` corresponds to (A_k', B_k') 
    # where A_k' is the k-th smallest A value (potentially with duplicates).
    for k in range(N):
        # Get the A and B values for the current element
        Ak = pairs[k][0] # This Ak serves as the maximum A constraint for this step
        Bk = pairs[k][1] # The B value of the current element

        # Add the current B value to our set of potential candidates for the K smallest B's.
        # Push -Bk onto the min-heap to maintain the max-heap property for B values.
        heapq.heappush(SmallK, -Bk)
        # Update the sum of B values in our current set of K smallest.
        current_sum_B += Bk

        # We only want to keep the K smallest B values. If the heap size exceeds K,
        # it means we've added an element that potentially makes the current K+1 elements.
        # We must remove the element with the largest B value among these K+1 to keep only K smallest.
        # The largest B value corresponds to the smallest negative value (-B_max), which is the root of the min-heap.
        if len(SmallK) > K:
            # Remove the element with the largest B value. heappop returns the smallest item (-B_max).
            largest_B_neg = heapq.heappop(SmallK)
            # Convert the popped negative value back to the actual B value.
            largest_B = -largest_B_neg
            # Adjust the sum by subtracting the removed largest B value.
            current_sum_B -= largest_B 

        # After processing the k-th element (0-indexed), we have considered k+1 elements.
        # If we have considered at least K elements (i.e., k+1 >= K, which means k >= K-1),
        # the heap `SmallK` contains exactly K elements, representing the K smallest B values
        # among the elements processed so far (those with A_i <= Ak).
        # We can then calculate a candidate value for the minimum product.
        if k >= K - 1:
            # The current Ak is the maximum A value among the first k+1 elements due to sorting.
            # This Ak acts as the constraint X = max_{i in S} A_i in this iteration.
            # current_sum_B is the sum of the K smallest B values among elements with A_i <= Ak.
            # Calculate the product: (max A constraint) * (sum of K smallest B's under this constraint)
            candidate_product = Ak * current_sum_B
            # Update the overall minimum product found across all considered Ak constraints.
            min_product = min(min_product, candidate_product)

    # After iterating through all N elements, min_product holds the minimum possible value.
    print(min_product)

# Read the number of test cases from standard input.
T = int(sys.stdin.readline())
# Process each test case.
for _ in range(T):
    solve()