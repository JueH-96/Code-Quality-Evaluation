import sys
import heapq

def main():
    input = sys.stdin.readline
    N, M = map(int, input().split())
    
    # Heap of currently-present people, by their original index
    avail = list(range(1, N+1))
    heapq.heapify(avail)
    
    # Heap of scheduled returns: (return_time, person_id)
    returns = []
    
    # Accumulated noodles per person (1-indexed)
    ans = [0] * (N+1)
    
    for _ in range(M):
        T, W, S = map(int, input().split())
        
        # Process all returns up to time T (inclusive)
        while returns and returns[0][0] <= T:
            rt, pid = heapq.heappop(returns)
            heapq.heappush(avail, pid)
        
        # If someone is available, give them W noodles and schedule their return
        if avail:
            pid = heapq.heappop(avail)
            ans[pid] += W
            heapq.heappush(returns, (T + S, pid))
        # else: no one gets these noodles
    
    # Output results
    out = sys.stdout.write
    for i in range(1, N+1):
        out(str(ans[i]) + "
")

if __name__ == "__main__":
    main()