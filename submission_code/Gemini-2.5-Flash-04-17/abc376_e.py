import heapq
import sys

# Increase recursion depth for potentially deep sorting (though not expected here)
# sys.setrecursionlimit(2000)

def solve():
    # Read N and K
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    K = int(line1[1])

    # Read A and B sequences
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    # Create pairs (A_i, B_i)
    items = []
    for i in range(N):
        items.append((A[i], B[i]))

    # Sort pairs based on A values
    # This is the core idea: consider sets S where max A is from a sorted prefix
    items.sort()

    # Initialize minimum product found so far to infinity
    min_total_product = float('inf')

    # Max-priority queue to keep track of the K smallest B values encountered among
    # items processed so far. We use a min-heap storing negative B values.
    pq = []
    # Variable to store the sum of B values currently in the priority queue
    current_sum_B = 0

    # Iterate through the sorted pairs
    for j in range(N):
        a_j, b_j = items[j]

        # Add the current item's B value to the priority queue
        # Use -b_j to make it a max-priority queue for B values
        heapq.heappush(pq, -b_j)
        current_sum_B += b_j

        # If the priority queue size exceeds K, remove the element with the largest B value
        # (which is the smallest negative value in the min-heap)
        if len(pq) > K:
            largest_b = -heapq.heappop(pq)
            current_sum_B -= largest_b

        # If the priority queue contains exactly K elements, we have a candidate set S.
        # This set S consists of the K elements with the smallest B values among the first j+1
        # pairs in the A-sorted list (items[0]...items[j]).
        if len(pq) == K:
            # The sum of B values for this set S is current_sum_B.
            # The maximum A value for this set S is items[j][0] (which is a_j).
            # This is because items are sorted by A. All items from index 0 to j have
            # A values less than or equal to a_j. The K items selected by the PQ are
            # chosen from these first j+1 items. The item items[j] has the largest
            # A value (a_j) among these first j+1 items. If items[j] is included in the
            # set of K smallest B values, then a_j is in the set and is the maximum A.
            # If items[j] is NOT included (meaning b_j was large and it was popped),
            # then all K selected items come from items[0]...items[j-1], whose max A
            # is <= items[j-1][0]. However, the logic holds because the optimal set S must
            # be the K smallest B values among {i | A_i <= V} where V = max_{k in S} A_k.
            # This loop structure considers V = items[j][0] for each j where PQ size reaches K.
            # When PQ size is K, the set of K smallest B values from items[0]...items[j]
            # is considered. The maximum A value of this set is indeed items[j][0].

            current_product = a_j * current_sum_B
            min_total_product = min(min_total_product, current_product)

    # Print the minimum total product found
    sys.stdout.write(str(min_total_product) + '
')


# Read the number of test cases
T = int(sys.stdin.readline())

# Process each test case
for _ in range(T):
    solve()