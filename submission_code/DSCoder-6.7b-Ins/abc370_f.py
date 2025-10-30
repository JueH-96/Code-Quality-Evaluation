import sys
from heapq import *

def solve():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    # Create a max heap for the masses of the pieces
    heap = [-A[i] for i in range(N)]
    heapify(heap)

    # Keep track of the total mass of the pieces
    total = sum(A)

    # Keep track of the minimum mass of the pieces
    min_mass = float('inf')

    # Keep track of the number of cut lines that are never cut
    cut_lines = 0

    # Distribute the pieces among the people
    for _ in range(K - 1):
        # Get the two largest pieces
        a = -heappop(heap)
        b = -heappop(heap)

        # Update the minimum mass
        min_mass = min(min_mass, a, b)

        # Update the total mass
        total -= a + b

        # If the two largest pieces are consecutive, increment the number of cut lines
        if len(heap) > 0 and -heap[0] == b - A[0]:
            cut_lines += 1

        # Add the remaining mass of the two largest pieces to the heap
        heappush(heap, -(a + b - total))

    # Update the minimum mass
    min_mass = min(min_mass, -heap[0])

    # Print the minimum mass and the number of cut lines
    print(min_mass, cut_lines)

solve()