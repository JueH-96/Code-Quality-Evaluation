import heapq
import sys

# The problem constraints on N and M are up to 2 * 10^5,
# and A_i, B_i up to 10^9.
# The total cost can exceed the capacity of a 32-bit integer,
# but Python's integers handle arbitrary size, so total_cost is safe.
# The time complexity will be dominated by sorting and heap operations,
# which is O(N log N + M log M + M log N), simplifying to O((N+M) log N) or O(N log N) since M <= N.
# This is efficient enough for the constraints.
# The space complexity is O(N + M) for storing A, B, and the heap.

def solve():
    # Read N (number of boxes) and M (number of people)
    N, M = map(int, sys.stdin.readline().split())

    # Read A (costs/candy counts)
    # A[i] is the price of box i+1 and the candy count in box i+1.
    # The list is 0-indexed internally.
    A = list(map(int, sys.stdin.readline().split()))

    # Read B (required candy counts)
    # B[i] is the minimum candy required by person i+1.
    # The list is 0-indexed internally.
    B = list(map(int, sys.stdin.readline().split()))

    # Sort requirements B in descending order.
    # This greedy approach processes the most difficult requirements first.
    B.sort(reverse=True)

    # Sort boxes A (candy/cost) in descending order by candy count (which is A_i).
    # This helps in efficiently identifying boxes that become eligible as the
    # required candy threshold decreases when iterating through sorted B.
    A.sort(reverse=True)

    # Variable to accumulate the minimum total cost.
    # Python's integers grow dynamically, so they handle large sums.
    total_cost = 0

    # Min-priority queue (min-heap) to store the costs of available boxes
    # that currently meet the candy requirement of the person being considered.
    # We use heapq which is Python's standard library for heap queues.
    available_costs = []
    # No need to call heapq.heapify on an initially empty list; push operations build the heap.

    # Pointer to track the index in the sorted A list from which we are
    # currently considering adding boxes to the 'available_costs' heap.
    box_idx = 0

    # Iterate through the sorted requirements B, from the highest requirement to the lowest.
    for req in B:
        # For the current requirement 'req', add all boxes from the sorted A list
        # (starting from box_idx) that have a candy count greater than or equal to 'req'
        # into the 'available_costs' heap. These are the boxes that are potentially
        # available and suitable for the current person (and potentially subsequent people
        # with lower requirements).
        while box_idx < N and A[box_idx] >= req:
            # Add the cost (which is A[box_idx]) to the min-heap.
            heapq.heappush(available_costs, A[box_idx])
            box_idx += 1

        # After considering all newly eligible boxes for the current requirement,
        # we need to select one box to satisfy this person.
        # The set of available and eligible boxes is represented by the costs in 'available_costs'.
        # If the heap is empty, it means there are no suitable boxes left.
        # Thus, it's impossible to satisfy the current requirement and all M requirements.
        if not available_costs:
            print(-1)
            return

        # According to the greedy strategy, to minimize the total cost,
        # we select the cheapest available box that meets the requirement.
        # The minimum element in the min-heap is the cheapest cost.
        min_cost = heapq.heappop(available_costs)

        # Add the cost of the selected box to the total accumulated cost.
        total_cost += min_cost

    # If the loop completes successfully for all M requirements, it means we have
    # found M distinct boxes satisfying all conditions. The accumulated 'total_cost'
    # is the minimum possible due to the greedy selection at each step.
    print(total_cost)

# Execute the solve function to run the program.
solve()