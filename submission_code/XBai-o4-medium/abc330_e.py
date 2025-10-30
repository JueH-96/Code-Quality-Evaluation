import heapq
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    
    max_val = N
    count = [0] * (max_val + 1)
    
    for num in A:
        if num <= max_val:
            count[num] += 1
    
    heap = [x for x in range(max_val + 1) if count[x] == 0]
    heapq.heapify(heap)
    
    for _ in range(Q):
        i = int(input[ptr]) - 1  # convert to 0-based index
        ptr += 1
        x = int(input[ptr])
        ptr += 1
        
        old_val = A[i]
        new_val = x
        
        # Process old_val
        if old_val <= max_val:
            count[old_val] -= 1
            if count[old_val] == 0:
                heapq.heappush(heap, old_val)
        
        # Process new_val
        if new_val <= max_val:
            if count[new_val] == 0:
                heapq.heappush(heap, new_val)
            count[new_val] += 1
        
        A[i] = new_val
        
        # Clean up the heap
        while heap and count[heap[0]] > 0:
            heapq.heappop(heap)
        
        print(heap[0])

if __name__ == "__main__":
    main()