# YOUR CODE HERE
import sys
import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    X = int(input[1])
    Y = int(input[2])
    A = list(map(int, input[3:3+N]))
    B = list(map(int, input[3+N:3+2*N]))

    # Create a list of dishes with their sweetness and saltiness
    dishes = [(A[i], B[i]) for i in range(N)]

    # Sort dishes by the ratio of sweetness to saltiness
    dishes.sort(key=lambda x: x[0] / x[1])

    # Use a min-heap to keep track of the dishes we have eaten
    heap = []
    total_sweetness = 0
    total_saltiness = 0
    min_dishes = float('inf')

    for i in range(N):
        # Add the current dish to the heap
        heapq.heappush(heap, (-dishes[i][0], -dishes[i][1]))
        total_sweetness += dishes[i][0]
        total_saltiness += dishes[i][1]

        # Check if we have exceeded the limits
        while total_sweetness > X or total_saltiness > Y:
            # Remove the dish with the highest sweetness to saltiness ratio
            removed_sweetness, removed_saltiness = heapq.heappop(heap)
            total_sweetness += removed_sweetness
            total_saltiness += removed_saltiness

        # Update the minimum number of dishes
        if total_sweetness <= X and total_saltiness <= Y:
            min_dishes = min(min_dishes, len(heap))

    print(min_dishes)

if __name__ == "__main__":
    main()