from heapq import heappop, heappush

def read_input():
    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    return N, K, A

def min_weirdness(N, K, A):
    # Initialize a min-heap with all pairs of socks
    heap = []
    for i in range(1, N + 1):
        if i not in A:
            heappush(heap, (i, i))
    
    # If there's an odd number of socks, remove one sock
    if (2 * N - K) % 2 == 1:
        heappop(heap)
    
    # Pair the socks to minimize weirdness
    total_weirdness = 0
    while len(heap) > 1:
        _, color1 = heappop(heap)
        _, color2 = heappop(heap)
        total_weirdness += abs(color1 - color2)
        # Push a dummy value to maintain the heap size
        heappush(heap, (0, 0))
    
    return total_weirdness

def main():
    N, K, A = read_input()
    result = min_weirdness(N, K, A)
    print(result)

if __name__ == "__main__":
    main()