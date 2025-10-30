import sys
import heapq

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N = int(input[ptr])
        K = int(input[ptr+1])
        ptr += 2
        A = list(map(int, input[ptr:ptr+N]))
        ptr += N
        B = list(map(int, input[ptr:ptr+N]))
        ptr += N
        
        # Sort the pairs by A in increasing order
        arr = sorted(zip(A, B), key=lambda x: x[0])
        
        heap = []
        sum_b = 0
        min_val = float('inf')
        
        for a, b in arr:
            heapq.heappush(heap, -b)
            sum_b += b
            
            if len(heap) > K:
                popped = -heapq.heappop(heap)
                sum_b -= popped
            
            if len(heap) == K:
                candidate = a * sum_b
                if candidate < min_val:
                    min_val = candidate
        
        print(min_val)

if __name__ == "__main__":
    main()