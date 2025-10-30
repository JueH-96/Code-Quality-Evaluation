import sys
import heapq

def main():
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    Q = int(data[ptr])
    ptr += 1
    A = list(map(int, data[ptr:ptr+N]))
    ptr += N

    max_x = N
    freq = [0] * (max_x + 2)
    for x in A:
        if x <= max_x:
            freq[x] += 1

    heap = []
    for x in range(max_x + 1):
        if freq[x] == 0:
            heapq.heappush(heap, x)

    for _ in range(Q):
        i = int(data[ptr]) - 1  # converting to 0-based index
        ptr += 1
        x = int(data[ptr])
        ptr += 1

        old_val = A[i]
        if old_val <= max_x:
            freq[old_val] -= 1
            if freq[old_val] == 0:
                heapq.heappush(heap, old_val)
        
        new_val = x
        A[i] = new_val
        if new_val <= max_x:
            if freq[new_val] == 0:
                # It was missing, now it's present. So, when we process the heap, it will be popped.
                pass
            freq[new_val] += 1
        
        # Process the heap to find mex
        while heap:
            current = heap[0]
            if freq[current] > 0:
                heapq.heappop(heap)
            else:
                break
        
        if not heap:
            mex = max_x + 1
        else:
            mex = heap[0]
        
        print(mex)

if __name__ == '__main__':
    main()