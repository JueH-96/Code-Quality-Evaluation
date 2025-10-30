import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    T = int(data[idx])
    idx += 1
    for _ in range(T):
        N, K = int(data[idx]), int(data[idx+1])
        idx +=2
        A = list(map(int, data[idx:idx+N]))
        idx +=N
        B = list(map(int, data[idx:idx+N]))
        idx +=N
        AB = list(zip(A, B))
        AB.sort()
        heap = []
        current_sum = 0
        min_product = float('inf')
        for a, b in AB:
            heapq.heappush(heap, -b)
            current_sum += b
            if len(heap) > K:
                removed = -heapq.heappop(heap)
                current_sum -= removed
            if len(heap) == K:
                product = a * current_sum
                if product < min_product:
                    min_product = product
        print(min_product)
        
if __name__ == "__main__":
    main()