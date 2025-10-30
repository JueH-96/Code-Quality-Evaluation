def solve():
    import sys
    import heapq

    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    A = list(map(int, data[2:2 + N]))
    B = list(map(int, data[2 + N:]))

    # Sort both lists in descending order
    A.sort(reverse=True)
    B.sort(reverse=True)

    # We'll use a min-heap to pick the cheapest box (smallest A_j) among those
    # that satisfy B_i (A_j >= B_i), as we proceed from largest B_i to smaller B_i.
    heap = []
    total_cost = 0
    j = 0  # Pointer for A

    # For each required candy count B_i in descending order:
    for b in B:
        # Add all boxes with A_j >= b to the min-heap
        while j < N and A[j] >= b:
            heapq.heappush(heap, A[j])
            j += 1

        # If we cannot find any box that satisfies b, it's impossible
        if not heap:
            print(-1)
            return

        # Pop the cheapest valid box from the heap
        total_cost += heapq.heappop(heap)

    # If we managed to assign a box for each B_i, print the total minimal cost
    print(total_cost)

# For testing in the same code environment
if __name__ == "__main__":
    solve()