import sys
import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    # Read parents for 2 to N
    parents = list(map(int, data[idx:idx + N-1]))
    idx += N-1
    
    # Build children list
    children = [[] for _ in range(N + 1)]
    for i in range(2, N+1):
        parent = parents[i-2]
        children[parent].append(i)
    
    # Read M insurance events
    cover = [ - (1 << 60) ] * (N + 1)
    for _ in range(M):
        x = int(data[idx])
        idx += 1
        y = int(data[idx])
        idx += 1
        if y > cover[x]:
            cover[x] = y
    
    # Build priority queue
    h = []
    for i in range(1, N+1):
        if cover[i] >= 0:
            heapq.heappush(h, (-cover[i], i))
    
    # Process queue
    while h:
        neg_cov, u = heapq.heappop(h)
        current_cov = -neg_cov
        if current_cov < cover[u]:
            continue
        # Propagate to children
        for v in children[u]:
            new_val = cover[u] - 1
            if new_val > cover[v]:
                cover[v] = new_val
                heapq.heappush(h, (-cover[v], v))
    
    # Count covered nodes
    ans = 0
    for i in range(1, N+1):
        if cover[i] >= 0:
            ans += 1
    print(ans)

if __name__ == "__main__":
    main()