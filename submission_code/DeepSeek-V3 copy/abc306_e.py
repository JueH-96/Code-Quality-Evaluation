import sys
import heapq

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    K = int(data[idx+1])
    Q = int(data[idx+2])
    idx += 3
    updates = []
    for _ in range(Q):
        X = int(data[idx])
        Y = int(data[idx+1])
        updates.append((X, Y))
        idx += 2
    A = [0] * (N + 1)
    # To manage the top K elements, we use a min-heap and a max-heap
    # min_heap will store the K largest elements
    # max_heap will store the rest
    min_heap = []
    max_heap = []
    # Initialize the heaps
    for i in range(1, N+1):
        heapq.heappush(max_heap, -A[i])
    # Move the K largest elements to min_heap
    for _ in range(K):
        if max_heap:
            val = -heapq.heappop(max_heap)
            heapq.heappush(min_heap, val)
    # Now, min_heap contains the K largest elements
    # We need to keep track of the sum of these K elements
    sum_k = sum(min_heap)
    for x, y in updates:
        old_val = A[x]
        A[x] = y
        # First, we need to remove the old value from the heaps
        # Since we don't know which heap it is in, we need to manage it
        # We can assume that if old_val is in min_heap, it is one of the K largest
        # Otherwise, it is in max_heap
        # To find out, we can check if old_val is >= the smallest in min_heap
        if min_heap and old_val >= min_heap[0]:
            # It is in min_heap
            # Remove it from min_heap
            # Since we can't directly remove, we need to manage it
            # We can mark it as to be removed and adjust the sum
            # But for simplicity, we can rebuild the heaps
            # Rebuild the heaps
            min_heap = []
            max_heap = []
            for i in range(1, N+1):
                heapq.heappush(max_heap, -A[i])
            for _ in range(K):
                if max_heap:
                    val = -heapq.heappop(max_heap)
                    heapq.heappush(min_heap, val)
            sum_k = sum(min_heap)
        else:
            # It is in max_heap
            # Remove it from max_heap
            # Since we can't directly remove, we need to manage it
            # Rebuild the heaps
            min_heap = []
            max_heap = []
            for i in range(1, N+1):
                heapq.heappush(max_heap, -A[i])
            for _ in range(K):
                if max_heap:
                    val = -heapq.heappop(max_heap)
                    heapq.heappush(min_heap, val)
            sum_k = sum(min_heap)
        # Now, we need to insert the new value y
        if min_heap and y >= min_heap[0]:
            # It should be in min_heap
            heapq.heappush(min_heap, y)
            sum_k += y
            # If min_heap has more than K elements, move the smallest to max_heap
            if len(min_heap) > K:
                val = heapq.heappop(min_heap)
                sum_k -= val
                heapq.heappush(max_heap, -val)
        else:
            # It should be in max_heap
            heapq.heappush(max_heap, -y)
            # If max_heap has elements larger than the smallest in min_heap, move them
            if max_heap and min_heap and -max_heap[0] > min_heap[0]:
                val = -heapq.heappop(max_heap)
                heapq.heappush(min_heap, val)
                sum_k += val
                # Move the smallest from min_heap to max_heap
                val = heapq.heappop(min_heap)
                sum_k -= val
                heapq.heappush(max_heap, -val)
        print(sum_k)

if __name__ == "__main__":
    main()