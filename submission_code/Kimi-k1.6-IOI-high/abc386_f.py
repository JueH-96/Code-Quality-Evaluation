import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    K = int(input[idx])
    idx += 1
    S = input[idx]
    idx += 1
    T = input[idx]
    idx += 1
    
    n = len(S)
    m = len(T)
    
    if abs(n - m) > K:
        print("No")
        return
    
    heap = []
    heapq.heappush(heap, (0, 0, 0))
    dist = {(0, 0): 0}
    found = False
    
    while heap:
        cost, i, j = heapq.heappop(heap)
        if i == n and j == m:
            found = True
            break
        if cost > K:
            continue
        current_dist = dist.get((i, j), float('inf'))
        if cost > current_dist:
            continue
        
        # Replace
        if i < n and j < m:
            new_cost = cost + (0 if S[i] == T[j] else 1)
            if new_cost <= K:
                new_i, new_j = i+1, j+1
                if dist.get((new_i, new_j), float('inf')) > new_cost:
                    dist[(new_i, new_j)] = new_cost
                    heapq.heappush(heap, (new_cost, new_i, new_j))
        
        # Delete
        if i < n:
            new_cost = cost + 1
            if new_cost <= K:
                new_i, new_j = i+1, j
                if dist.get((new_i, new_j), float('inf')) > new_cost:
                    dist[(new_i, new_j)] = new_cost
                    heapq.heappush(heap, (new_cost, new_i, new_j))
        
        # Insert
        if j < m:
            new_cost = cost + 1
            if new_cost <= K:
                new_i, new_j = i, j+1
                if dist.get((new_i, new_j), float('inf')) > new_cost:
                    dist[(new_i, new_j)] = new_cost
                    heapq.heappush(heap, (new_cost, new_i, new_j))
    
    print("Yes" if found else "No")

if __name__ == "__main__":
    main()