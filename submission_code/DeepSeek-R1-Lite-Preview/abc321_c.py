import heapq

def find_kth_321_like_number(K):
    heap = list(range(1, 10))  # Initialize heap with 1 to 9
    heapq.heapify(heap)
    
    while K > 1:
        current = heapq.heappop(heap)
        last_digit = current % 10
        for d in range(0, last_digit):
            next_number = current * 10 + d
            heapq.heappush(heap, next_number)
        K -= 1
    return heap[0]

if __name__ == "__main__":
    K = int(input())
    print(find_kth_321_like_number(K))