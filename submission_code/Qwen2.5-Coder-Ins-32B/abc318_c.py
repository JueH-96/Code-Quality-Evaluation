# YOUR CODE HERE
import sys

def main():
    import sys
    import heapq

    input = sys.stdin.read().split()
    N = int(input[0])
    D = int(input[1])
    P = int(input[2])
    F = list(map(int, input[3:]))

    # Calculate the minimum cost
    min_cost = 0
    heap = []

    for i in range(N):
        heapq.heappush(heap, F[i])
        if len(heap) > D:
            min_cost += heapq.heappop(heap)
        else:
            min_cost += F[i]

    # Calculate the cost of using passes
    num_passes = (N + D - 1) // D
    pass_cost = num_passes * P

    # The result is the minimum of using passes or paying for individual days
    result = min(min_cost, pass_cost)

    print(result)

if __name__ == "__main__":
    main()