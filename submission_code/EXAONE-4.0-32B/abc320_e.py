import heapq
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    m = int(data[1])
    
    events = []
    index = 2
    for i in range(m):
        T = int(data[index])
        W = int(data[index+1])
        S = int(data[index+2])
        index += 3
        events.append((T, W, S))
    
    available = list(range(1, n+1))
    heapq.heapify(available)
    returns = []
    ans = [0] * (n+1)
    
    for T, W, S in events:
        while returns and returns[0][0] <= T:
            return_time, person = heapq.heappop(returns)
            heapq.heappush(available, person)
            
        if available:
            p = heapq.heappop(available)
            ans[p] += W
            heapq.heappush(returns, (T + S, p))
            
    for i in range(1, n+1):
        print(ans[i])

if __name__ == "__main__":
    main()