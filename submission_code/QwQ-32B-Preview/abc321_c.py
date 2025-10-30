import heapq

def find_kth_321_like_number(K):
    # Initialize a min-heap with 1-digit numbers
    heap = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    heapq.heapify(heap)
    
    for _ in range(K - 1):
        current = heapq.heappop(heap)
        last_digit = current % 10
        for next_digit in range(0, last_digit):
            new_number = current * 10 + next_digit
            heapq.heappush(heap, new_number)
    
    # The top of the heap is now the K-th smallest 321-like number
    return str(heapq.heappop(heap))

# Read K from input
K = int(input())
print(find_kth_321_like_number(K))