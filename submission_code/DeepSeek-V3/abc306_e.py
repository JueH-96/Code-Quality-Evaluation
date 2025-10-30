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
    
    A = [0] * (N + 1)  # 1-based indexing
    # To manage the top K elements, we use a min-heap for the top K and a max-heap for the rest
    # But since we need to efficiently manage the top K elements, we can use a min-heap for the top K and a max-heap for the rest
    # However, in Python, heapq is a min-heap by default, so to simulate a max-heap, we can insert negative values
    # So, we will maintain two heaps:
    # - min_heap: contains the top K elements (smallest among the top K is at the top)
    # - max_heap: contains the rest of the elements (largest among the rest is at the top)
    
    # Initialize the heaps
    min_heap = []  # contains the top K elements
    max_heap = []  # contains the rest
    
    # Initially, all elements are 0, so the top K are the first K zeros
    for i in range(1, K+1):
        heapq.heappush(min_heap, 0)
    for i in range(K+1, N+1):
        heapq.heappush(max_heap, -0)
    
    # To keep track of the elements in the heaps, we can use a dictionary
    # But since the elements are not unique, we need to manage the counts
    # So, we will use a dictionary to count the occurrences of each element in the heaps
    # But for simplicity, we can manage the counts separately for min_heap and max_heap
    # However, it's easier to manage the counts in a single dictionary and adjust the heaps accordingly
    
    # Instead, we can manage the counts in a dictionary and adjust the heaps accordingly
    # But for the purpose of this problem, since the elements are not unique, we need to manage the counts
    # So, we will use a dictionary to count the occurrences of each element in the heaps
    
    # Initialize the counts
    count = {}
    for i in range(1, N+1):
        count[i] = 0
    
    # Now, process the updates
    for X, Y in updates:
        # First, remove the old value from the heaps
        old_val = A[X]
        if old_val in count:
            count[old_val] -= 1
            if count[old_val] == 0:
                del count[old_val]
        
        # Update the array
        A[X] = Y
        # Add the new value to the heaps
        if Y in count:
            count[Y] += 1
        else:
            count[Y] = 1
        
        # Now, we need to adjust the heaps to maintain the top K elements
        # First, we need to ensure that the min_heap contains exactly K elements
        # If the size of min_heap is less than K, we need to move elements from max_heap to min_heap
        # If the size of min_heap is more than K, we need to move elements from min_heap to max_heap
        
        # First, we need to ensure that the min_heap contains exactly K elements
        while len(min_heap) < K and max_heap:
            # Move the largest element from max_heap to min_heap
            val = -heapq.heappop(max_heap)
            heapq.heappush(min_heap, val)
        while len(min_heap) > K:
            # Move the smallest element from min_heap to max_heap
            val = heapq.heappop(min_heap)
            heapq.heappush(max_heap, -val)
        
        # Now, we need to ensure that the top K elements are in min_heap
        # So, we need to check if the smallest element in min_heap is smaller than the largest element in max_heap
        # If so, we need to swap them
        if min_heap and max_heap and min_heap[0] < -max_heap[0]:
            # Swap the smallest in min_heap with the largest in max_heap
            min_val = heapq.heappop(min_heap)
            max_val = -heapq.heappop(max_heap)
            heapq.heappush(min_heap, max_val)
            heapq.heappush(max_heap, -min_val)
        
        # Now, calculate the sum of the top K elements
        sum_top_k = sum(min_heap)
        print(sum_top_k)

if __name__ == "__main__":
    main()