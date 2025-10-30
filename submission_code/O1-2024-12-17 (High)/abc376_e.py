def main():
    import sys
    import heapq
    
    input = sys.stdin.readline
    T = int(input())
    
    for _ in range(T):
        N, K = map(int, input().split())
        A = list(map(int, input().split()))
        B = list(map(int, input().split()))
        
        # Pair up A_i and B_i, then sort by A_i ascending
        pairs = sorted(zip(A, B), key=lambda x: x[0])
        
        # We will maintain a max-heap (via storing negative values) of size up to K 
        # to keep track of the K smallest B_i in the current prefix as we iterate.
        heap = []
        sum_b = 0
        min_cost = float('inf')
        
        for a, b in pairs:
            # Add b to the heap (store negative for max-heap effect)
            sum_b += b
            heapq.heappush(heap, -b)
            
            # If we exceed K elements, remove the largest b
            if len(heap) > K:
                largest = -heapq.heappop(heap)
                sum_b -= largest
            
            # Once we have exactly K elements, compute the cost
            if len(heap) == K:
                cost = a * sum_b
                if cost < min_cost:
                    min_cost = cost
        
        print(min_cost)

# Don't forget to call main() at the end
if __name__ == "__main__":
    main()