import sys, heapq

def main():
    import sys
    import sys
    from sys import stdin
    def input():
        return sys.stdin.read()
    data = input().split()
    idx = 0
    N = int(data[idx]); idx +=1
    adj = [[] for _ in range(N+1)]
    for i in range(1, N):
        A = int(data[idx]); idx +=1
        B = int(data[idx]); idx +=1
        X = int(data[idx]); idx +=1
        adj[i].append( (i+1, A) )
        adj[i].append( (X, B) )
    INF = 1<<60
    min_time = [INF]*(N+1)
    min_time[1] = 0
    heap = []
    heapq.heappush(heap, (0, 1))
    while heap:
        current_time, node = heapq.heappop(heap)
        if node == N:
            print(current_time)
            return
        if current_time > min_time[node]:
            continue
        for neighbor, cost in adj[node]:
            new_time = current_time + cost
            if new_time < min_time[neighbor]:
                min_time[neighbor] = new_time
                heapq.heappush(heap, (new_time, neighbor))
    print(min_time[N])

if __name__ == "__main__":
    main()