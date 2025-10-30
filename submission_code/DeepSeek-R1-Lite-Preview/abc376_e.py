import sys
import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    T = int(input[idx])
    idx += 1
    for _ in range(T):
        N = int(input[idx])
        K = int(input[idx + 1])
        idx += 2
        A = [int(input[idx + i]) for i in range(N)]
        idx += N
        B = [int(input[idx + i]) for i in range(N)]
        idx += N
        # Pair A and B, sort by A ascending
        sorted_pairs = sorted(zip(A, B), key=lambda x: x[0])
        heap = []
        sum_heap = 0
        min_product = float('inf')
        for i in range(N):
            a, b = sorted_pairs[i]
            heapq.heappush(heap, -b)
            sum_heap += b
            if len(heap) > K:
                removed_b = -heapq.heappop(heap)
                sum_heap -= removed_b
            if len(heap) == K:
                product = a * sum_heap
                if product < min_product:
                    min_product = product
        print(min_product)

if __name__ == "__main__":
    main()