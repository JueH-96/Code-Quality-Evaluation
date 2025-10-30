import heapq
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N, K = int(input[ptr]), int(input[ptr+1])
        ptr += 2
        A = list(map(int, input[ptr:ptr+N]))
        ptr += N
        B = list(map(int, input[ptr:ptr+N]))
        ptr += N
        
        sorted_list = sorted(zip(A, B), key=lambda x: x[0])
        heap = []
        current_sum = 0
        min_product = float('inf')
        
        for i, (a, b) in enumerate(sorted_list):
            if len(heap) < K:
                heapq.heappush(heap, -b)
                current_sum += b
            else:
                if b < -heap[0]:
                    current_sum += b + heap[0]  # because heap[0] is -max_b
                    heapq.heappop(heap)
                    heapq.heappush(heap, -b)
            
            if i + 1 >= K:
                if a * current_sum < min_product:
                    min_product = a * current_sum
        
        print(min_product)

if __name__ == "__main__":
    main()