# YOUR CODE HERE
import sys
import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    K = int(input[idx + 1])
    Q = int(input[idx + 2])
    idx += 3

    A = [0] * N
    heap = []
    for i in range(N):
        heapq.heappush(heap, (0, i))

    results = []
    for _ in range(Q):
        X = int(input[idx]) - 1
        Y = int(input[idx + 1])
        idx += 2

        old_val, _ = heap[A[X]]
        if old_val != A[X]:
            heap.remove((old_val, X))
            heapq.heapify(heap)

        A[X] = Y
        heapq.heappush(heap, (Y, X))

        if len(heap) > K:
            heapq.heappop(heap)

        top_k_sum = sum(val for val, _ in heap)
        results.append(top_k_sum)

    for result in results:
        print(result)

if __name__ == "__main__":
    main()