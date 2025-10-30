import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    idx += 1
    
    D = list(map(int, data[idx:idx+N]))
    idx += N
    
    L1, C1, K1 = map(int, data[idx:idx+3])
    idx +=3
    L2, C2, K2 = map(int, data[idx:idx+3])
    
    # Initialize the priority queue with (cost, k1, k2)
    heap = []
    heapq.heappush(heap, (0, 0, 0))
    current_best = {(0, 0): 0}
    
    for di in D:
        next_heap = []
        next_best = {}
        # Process all states in the current priority queue
        while heap:
            cost, k1, k2 = heapq.heappop(heap)
            # Skip if this state is not the best known
            if current_best.get((k1, k2), float('inf')) < cost:
                continue
            # Generate all possible a for this state
            max_a_possible = K1 - k1
            for a in range(0, max_a_possible + 1):
                remaining = di - a * L1
                if remaining <= 0:
                    b = 0
                else:
                    b = (remaining + L2 - 1) // L2
                new_k2 = k2 + b
                if new_k2 > K2:
                    continue
                new_k1 = k1 + a
                new_cost = cost + a * C1 + b * C2
                # Check if this new state is valid and better
                key = (new_k1, new_k2)
                if key not in next_best or new_cost < next_best[key]:
                    next_best[key] = new_cost
                    heapq.heappush(next_heap, (new_cost, new_k1, new_k2))
        # Prepare for next iteration
        current_best = next_best
        heap = next_heap
        if not heap:
            break
    
    if current_best:
        print(min(current_best.values()))
    else:
        print(-1)

if __name__ == "__main__":
    main()