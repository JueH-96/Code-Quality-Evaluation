import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx +=1
    M = int(data[idx])
    idx +=1
    
    available = list(range(1, N+1))
    heapq.heapify(available)
    unavailable = []
    res = [0] * (N +1)
    
    for _ in range(M):
        T = int(data[idx])
        W = int(data[idx+1])
        S = int(data[idx+2])
        idx +=3
        
        # Move all people who have returned by time T back to available
        while unavailable and unavailable[0][0] <= T:
            rt, p = heapq.heappop(unavailable)
            heapq.heappush(available, p)
        
        if available:
            p = heapq.heappop(available)
            res[p] += W
            heapq.heappush(unavailable, (T + S, p))
    
    for i in range(1, N+1):
        print(res[i])

if __name__ == "__main__":
    main()