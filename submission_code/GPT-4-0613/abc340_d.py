import sys
from heapq import heappop, heappush

def main():
    N = int(sys.stdin.readline())
    stages = [list(map(int, sys.stdin.readline().split())) for _ in range(N-1)]
    stages.append([0, 0, 0])
    dp = [float('inf')] * (N+1)
    dp[1] = 0
    heap = [(0, 1)]
    while heap:
        time, stage = heappop(heap)
        if dp[stage] < time:
            continue
        A, B, X = stages[stage-1]
        if dp[stage+1] > time + A:
            dp[stage+1] = time + A
            heappush(heap, (dp[stage+1], stage+1))
        if dp[X] > time + B:
            dp[X] = time + B
            heappush(heap, (dp[X], X))
    print(dp[N])

if __name__ == "__main__":
    main()