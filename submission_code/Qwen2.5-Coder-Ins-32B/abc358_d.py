# YOUR CODE HERE
import sys
import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    A = list(map(int, input[2:2+N]))
    B = list(map(int, input[2+N:2+N+M]))

    # Create a list of tuples (A_i, B_i) and sort by B_i in descending order
    boxes = [(A[i], B[i]) for i in range(N)]
    boxes.sort(key=lambda x: x[1], reverse=True)

    # Use a min-heap to keep track of the M smallest A_i values that satisfy B_i
    min_heap = []
    total_cost = 0
    satisfied = 0

    for box in boxes:
        if box[0] >= box[1]:
            heapq.heappush(min_heap, box[0])
            total_cost += box[0]
            satisfied += 1

            if satisfied > M:
                total_cost -= heapq.heappop(min_heap)
                satisfied -= 1

        if satisfied == M:
            break

    if satisfied < M:
        print(-1)
    else:
        print(total_cost)

if __name__ == "__main__":
    main()