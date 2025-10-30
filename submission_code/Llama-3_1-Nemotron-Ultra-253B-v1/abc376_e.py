import heapq
import sys

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
        combined = sorted(zip(A, B), key=lambda x: x[0])
        heap = []
        current_sum = 0
        min_product = float('inf')
        for a_i, b_i in combined:
            heapq.heappush(heap, -b_i)
            current_sum += b_i
            if len(heap) > K:
                removed = -heapq.heappop(heap)
                current_sum -= removed
            if len(heap) == K:
                product = current_sum * a_i
                if product < min_product:
                    min_product = product
        print(min_product)

if __name__ == "__main__":
    main()