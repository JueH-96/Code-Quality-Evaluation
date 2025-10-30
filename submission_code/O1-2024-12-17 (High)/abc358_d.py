def main():
    import sys
    import heapq

    input = sys.stdin.readline

    # Read inputs
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))

    # Sort box capacities (and costs, since they are equal) in descending order
    A.sort(reverse=True)
    # Sort required candies in descending order
    B.sort(reverse=True)

    # We'll use a min-heap (priority queue) to select the cheapest available box
    # that can satisfy each person's candy requirement
    min_heap = []
    total_cost = 0
    index = 0  # index to iterate over A

    for required_candies in B:
        # Add all boxes that can satisfy 'required_candies' into the min-heap
        while index < N and A[index] >= required_candies:
            heapq.heappush(min_heap, A[index])
            index += 1

        # If no suitable box is available, it's impossible
        if not min_heap:
            print(-1)
            return

        # Choose the cheapest box that meets the requirement
        total_cost += heapq.heappop(min_heap)

    # If we can assign M boxes successfully, print the total cost
    print(total_cost)

# Do not forget to call main!
if __name__ == "__main__":
    main()