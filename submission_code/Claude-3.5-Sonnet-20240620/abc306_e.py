# YOUR CODE HERE
import heapq

def main():
    N, K, Q = map(int, input().split())
    
    # Initialize the sequence and heap
    sequence = [0] * N
    heap = []
    
    for _ in range(Q):
        X, Y = map(int, input().split())
        X -= 1  # Convert to 0-based index
        
        old_value = sequence[X]
        sequence[X] = Y
        
        # Update the heap
        if len(heap) < K:
            heapq.heappush(heap, Y)
        elif Y > heap[0]:
            heapq.heapreplace(heap, Y)
        
        # If the old value was in the heap, remove it and add the new value
        if old_value in heap:
            heap.remove(old_value)
            heapq.heapify(heap)
            if len(heap) < K and Y not in heap:
                heapq.heappush(heap, Y)
        
        # Calculate and print f(A)
        print(sum(heap))

if __name__ == "__main__":
    main()