import sys
import heapq

def main():
    sys.setrecursionlimit(1 << 25)
    n = int(sys.stdin.readline())
    edges = [[] for _ in range(n)]
    for _ in range(n-1):
        u, v = map(int, sys.stdin.readline().split())
        edges[u].append(v)
        edges[v].append(u)
    
    def compute_times(start):
        time = [float('inf')] * n
        time[start] = 0
        heap = []
        heapq.heappush(heap, (0, start))
        
        while heap:
            t, u = heapq.heappop(heap)
            if t > time[u]:
                continue
            for v in edges[u]:
                if v == start:
                    continue
                if t + (1 if v % 2 else 2) < time[v]:
                    time[v] = t + (1 if v % 2 else 2)
                    heapq.heappush(heap, (time[v], v))
        return time
    
    times = []
    for i in range(n):
        t = compute_times(i)
        max_time = max(t)
        times.append(max_time)
    
    print(' '.join(map(str, times)))

if __name__ == '__main__':
    main()