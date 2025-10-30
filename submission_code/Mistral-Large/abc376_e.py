import sys
import heapq

def solve():
    input = sys.stdin.read
    data = input().split()

    index = 0
    T = int(data[index])
    index += 1
    results = []

    for _ in range(T):
        N = int(data[index])
        K = int(data[index + 1])
        index += 2
        A = list(map(int, data[index:index + N]))
        index += N
        B = list(map(int, data[index:index + N]))
        index += N

        # Combine A and B into a list of tuples and sort by A in descending order
        AB = list(zip(A, B))
        AB.sort(key=lambda x: -x[0])

        # Use a min-heap to keep track of the smallest K elements based on B
        heap = []
        sum_B = 0
        max_A = 0

        for a, b in AB:
            if len(heap) < K:
                heapq.heappush(heap, (b, a))
                sum_B += b
                max_A = a
            else:
                if b < heap[0][0]:
                    sum_B -= heapq.heappop(heap)[0] - b
                    heapq.heappush(heap, (b, a))
                    sum_B += b
                    max_A = a

        results.append(max_A * sum_B)

    sys.stdout.write("
".join(map(str, results)) + "
")

if __name__ == "__main__":
    solve()