import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    W = int(data[1])
    items = []
    idx = 2
    for _ in range(N):
        w = int(data[idx])
        v = int(data[idx+1])
        items.append((w, v))
        idx += 2
    
    # Initialize priority queue with negative marginal happiness per weight
    # Start with k=0 for each type
    pq = []
    for i in range(N):
        w, v = items[i]
        if v - 0 > 0:
            marginal_per_weight = (v - 1) / w
            heapq.heappush(pq, (-marginal_per_weight, i, 0))
    
    total_happiness = 0
    total_weight = 0
    k = [0] * N  # to keep track of current k_i for each type
    
    while pq and total_weight + items[pq[0][1]][0] <= W:
        _, i, current_k = heapq.heappop(pq)
        w, v = items[i]
        if current_k >= 0:
            if total_weight + w <= W:
                # Add one item of this type
                marginal_happiness = v - 2 * current_k
                total_happiness += marginal_happiness
                total_weight += w
                k[i] += 1
                # Push the next marginal happiness if positive
                next_marginal = v - 2 * k[i]
                if next_marginal > 0:
                    next_marginal_per_weight = next_marginal / w
                    heapq.heappush(pq, (-next_marginal_per_weight, i, k[i]))
    
    print(total_happiness)

if __name__ == '__main__':
    main()