import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1

    events = []
    for _ in range(M):
        T = int(input[idx])
        idx += 1
        W = int(input[idx])
        idx += 1
        S = int(input[idx])
        idx += 1
        events.append((T, W, S))
    
    available = list(range(1, N+1))
    heapq.heapify(available)
    
    busy = []
    totals = [0] * (N + 1)  # 1-based indexing
    
    for t, w, s in events:
        # Process all returned people whose return time <= current t
        while busy and busy[0][0] <= t:
            return_time, p = heapq.heappop(busy)
            heapq.heappush(available, p)
        
        if available:
            p = heapq.heappop(available)
            totals[p] += w
            heapq.heappush(busy, (t + s, p))
    
    for i in range(1, N+1):
        print(totals[i])

if __name__ == "__main__":
    main()