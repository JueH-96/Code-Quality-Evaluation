import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    T = int(data[idx])
    idx += 1
    for _ in range(T):
        N = int(data[idx])
        K = int(data[idx+1])
        idx += 2
        A = list(map(int, data[idx:idx+N]))
        idx += N
        B = list(map(int, data[idx:idx+N]))
        idx += N
        sorted_pairs = sorted(zip(A, B), key=lambda x: x[0])
        heap = []
        current_sum = 0
        min_product = float('inf')
        for a, b in sorted_pairs:
            heapq.heappush(heap, -b)
            current_sum += b
            while len(heap) > K:
                removed = -heapq.heappop(heap)
                current_sum -= removed
            if len(heap) == K:
                product = current_sum * a
                if product < min_product:
                    min_product = product
        print(min_product)

if __name__ == "__main__":
    main()