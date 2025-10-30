import heapq
import sys
input = sys.stdin.read

def main():
    data = input().split()
    index = 0

    N = int(data[index])
    K = int(data[index + 1])
    Q = int(data[index + 2])
    index += 3

    A = [0] * N
    max_heap = []

    def update_heap(val, sign):
        heapq.heappush(max_heap, (-val, sign))
        if len(max_heap) > K:
            heapq.heappop(max_heap)

    for _ in range(Q):
        X = int(data[index]) - 1
        Y = int(data[index + 1])
        index += 2

        if A[X] != 0:
            update_heap(A[X], -1)

        A[X] = Y
        update_heap(Y, 1)

        current_sum = sum(-val for val, sign in max_heap)
        print(current_sum)

if __name__ == "__main__":
    main()