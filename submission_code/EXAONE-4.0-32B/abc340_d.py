import heapq
import sys

def main():
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    stages = []
    for i in range(1, n):
        parts = data[i].split()
        A = int(parts[0])
        B = int(parts[1])
        X = int(parts[2])
        stages.append((A, B, X))
    
    INF = 10**18
    dp = [INF] * (n + 1)
    dp[1] = 0
    heap = [(0, 1)]
    
    while heap:
        t, u = heapq.heappop(heap)
        if t != dp[u]:
            continue
        if u == n:
            break
        A, B, X = stages[u - 1]
        next_node1 = u + 1
        new_time1 = t + A
        if new_time1 < dp[next_node1]:
            dp[next_node1] = new_time1
            heapq.heappush(heap, (new_time1, next_node1))
        
        next_node2 = X
        new_time2 = t + B
        if new_time2 < dp[next_node2]:
            dp[next_node2] = new_time2
            heapq.heappush(heap, (new_time2, next_node2))
    
    print(dp[n])

if __name__ == "__main__":
    main()