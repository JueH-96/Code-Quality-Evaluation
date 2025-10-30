import heapq

def read_ints():
    return list(map(int, input().split()))

def main():
    N, M = read_ints()
    A = read_ints()
    B = read_ints()

    # Sort the boxes by price in ascending order
    boxes = sorted(zip(A, range(N)), key=lambda x: x[0])

    # Sort the people by the number of candies they need in descending order
    people = sorted(zip(B, range(M)), key=lambda x: x[0], reverse=True)

    # Use a min-heap to keep track of the cheapest boxes that can satisfy the current person's need
    available_boxes = []
    total_cost = 0
    j = 0

    for candies_needed, _ in people:
        # Add all boxes that can satisfy the current person's need to the heap
        while j < N and boxes[j][0] >= candies_needed:
            heapq.heappush(available_boxes, boxes[j])
            j += 1

        # If there are no available boxes that can satisfy the need, it's impossible
        if not available_boxes:
            print(-1)
            return

        # Choose the cheapest available box
        cost, _ = heapq.heappop(available_boxes)
        total_cost += cost

    print(total_cost)

if __name__ == "__main__":
    main()