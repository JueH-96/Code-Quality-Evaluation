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
        # Pair A and B together and sort based on A
        pairs = list(zip(A, B))
        pairs.sort()
        # Now, for each possible max A_i, select the smallest K B_i's where A_i <= max_A
        # We need to find the minimal (max_A * sum_B)
        # So, for each i from K-1 to N-1, consider pairs[i][0] as max_A
        # and select the smallest K B's from pairs[0..i]
        # To efficiently get the smallest K B's, we can use a max-heap of size K
        min_sum = float('inf')
        max_heap = []
        current_sum = 0
        for i in range(N):
            # Push B_i to the heap
            if len(max_heap) < K:
                heapq.heappush(max_heap, -pairs[i][1])
                current_sum += pairs[i][1]
            else:
                if pairs[i][1] < -max_heap[0]:
                    current_sum += pairs[i][1] + max_heap[0]
                    heapq.heappop(max_heap)
                    heapq.heappush(max_heap, -pairs[i][1])
            # When we have K elements, calculate the product
            if len(max_heap) == K:
                max_A = pairs[i][0]
                product = max_A * current_sum
                if product < min_sum:
                    min_sum = product
        print(min_sum)

if __name__ == "__main__":
    solve()