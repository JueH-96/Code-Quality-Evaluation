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
    
    events = []
    for _ in range(M):
        T = int(data[idx])
        idx += 1
        W = int(data[idx])
        idx += 1
        S = int(data[idx])
        idx += 1
        events.append((T, W, S))
    
    available = list(range(1, N + 1))
    heapq.heapify(available)
    
    returns = []
    result = [0] * N
    
    for t, w, s in events:
        while returns and returns[0][0] <= t:
            return_time, idx = heapq.heappop(returns)
            heapq.heappush(available, idx)
        
        if available:
            idx = heapq.heappop(available)
            result[idx - 1] += w
            heapq.heappush(returns, (t + s, idx))
    
    for res in result:
        print(res)

if __name__ == "__main__":
    main()