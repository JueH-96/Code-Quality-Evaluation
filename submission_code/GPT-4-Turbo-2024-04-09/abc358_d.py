def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:N+2]))
    B = list(map(int, data[N+2:N+2+M]))
    
    # Sort A and B to try to match the smallest requirements with the cheapest boxes
    A_sorted = sorted((price, price) for price in A)
    B_sorted = sorted((requirement, i) for i, requirement in enumerate(B))
    
    # Greedy matching from the least requirement to the highest
    import heapq
    min_heap = []
    j = 0
    total_cost = 0
    possible = True
    
    for requirement, _ in B_sorted:
        # Push all boxes that can satisfy the current requirement into the heap
        while j < N and A_sorted[j][0] >= requirement:
            heapq.heappush(min_heap, A_sorted[j])
            j += 1
        
        if min_heap:
            # Get the box with the minimum price that can satisfy the requirement
            price, _ = heapq.heappop(min_heap)
            total_cost += price
        else:
            # If no box can satisfy the requirement, it's impossible
            possible = False
            break
    
    if possible and len(min_heap) >= 0 and j == N and len(min_heap) == N - M:
        print(total_cost)
    else:
        print(-1)

if __name__ == "__main__":
    main()