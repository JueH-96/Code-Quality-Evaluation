import sys
import heapq

def solve():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    T = int(data[idx])
    idx += 1
    for _ in range(T):
        N, K = int(data[idx]), int(data[idx+1])
        idx += 2
        A = list(map(int, data[idx:idx+N]))
        idx += N
        B = list(map(int, data[idx:idx+N]))
        idx += N
        # Pair A and B together
        pairs = list(zip(A, B))
        # Sort based on A
        pairs.sort()
        # Initialize a max-heap to keep the smallest K B's
        # Since heapq is a min-heap, we store negative values
        heap = []
        sum_b = 0
        for i in range(K):
            sum_b += pairs[i][1]
            heapq.heappush(heap, -pairs[i][1])
        # Initialize the result
        res = pairs[K-1][0] * sum_b
        # Iterate from K to N-1
        for i in range(K, N):
            # Remove the largest B in the current K elements
            # Since it's a max-heap, we pop the smallest negative
            largest_b = -heapq.heappop(heap)
            sum_b -= largest_b
            # Add the new B
            sum_b += pairs[i][1]
            heapq.heappush(heap, -pairs[i][1])
            # Update the result
            current_max_a = pairs[i][0]
            current_sum_b = sum_b
            res = min(res, current_max_a * current_sum_b)
        print(res)

if __name__ == "__main__":
    solve()