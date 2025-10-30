import heapq
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N, K = map(int, input[ptr:ptr+2])
        ptr += 2
        A = list(map(int, input[ptr:ptr+N]))
        ptr += N
        B = list(map(int, input[ptr:ptr+N]))
        ptr += N
        
        sorted_pairs = sorted(zip(A, B), key=lambda x: x[0])
        A_sorted = [x[0] for x in sorted_pairs]
        B_sorted = [x[1] for x in sorted_pairs]
        
        if K == 1:
            min_product = min(a * b for a, b in zip(A_sorted, B_sorted))
        elif K == N:
            sum_B = sum(B_sorted)
            min_product = A_sorted[-1] * sum_B
        else:
            heap = []
            sum_heap = 0
            min_product = float('inf')
            for j in range(len(B_sorted)):
                current_B = B_sorted[j]
                heapq.heappush(heap, -current_B)
                sum_heap += current_B
                if len(heap) > K - 1:
                    removed = -heapq.heappop(heap)
                    sum_heap -= removed
                if j >= K - 2:
                    current_A = A_sorted[j + 1]
                    current_sum = sum_heap + B_sorted[j + 1]
                    product = current_A * current_sum
                    if product < min_product:
                        min_product = product
        print(min_product)

if __name__ == '__main__':
    main()