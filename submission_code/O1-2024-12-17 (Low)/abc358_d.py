def main():
    import sys
    import heapq

    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    A = list(map(int, input_data[2:2+N]))
    B = list(map(int, input_data[2+N:2+N+M]))

    # Sort boxes (capacities/costs) in descending order
    A.sort(reverse=True)
    # Sort demanded candies in descending order
    B.sort(reverse=True)

    # We'll go through each demanded candy count from largest to smallest,
    # and maintain a pointer into A. For those boxes that can satisfy the
    # current demand, we'll keep them in a min-heap so we can always pick
    # the box with the smallest cost among those that qualify.
    
    heap = []
    heapq.heapify(heap)
    total_cost = 0
    pointer_A = 0  # Pointer over A (which is sorted descending)
    for demand in B:
        # Move pointer_A forward while boxes can meet this demand
        while pointer_A < N and A[pointer_A] >= demand:
            # Add the box cost (capacity) to the min-heap
            heapq.heappush(heap, A[pointer_A])
            pointer_A += 1
        
        # If we have no available boxes to cover this demand, it's impossible
        if not heap:
            print(-1)
            return
        
        # Otherwise, choose the cheapest box (smallest A) among those >= demand
        chosen_box = heapq.heappop(heap)
        total_cost += chosen_box
    
    # If we manage to assign a box for each person, print the total cost
    print(total_cost)

# Do not forget to call main()
if __name__ == "__main__":
    main()