import sys
import bisect

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    graph = [[] for _ in range(n+1)]
    deg = [0] * (n+1)
    index = 1
    for _ in range(n-1):
        u = int(data[index])
        v = int(data[index+1])
        index += 2
        graph[u].append(v)
        graph[v].append(u)
        deg[u] += 1
        deg[v] += 1
    
    ans = n
    for c in range(1, n+1):
        neighbors = graph[c]
        k = len(neighbors)
        if k == 0:
            continue
        
        D = []
        for v in neighbors:
            D.append(deg[v] - 1)
        
        max_d = max(D)
        distinct_vals = set(D)
        candidate_ys = {1}
        for t in distinct_vals:
            candidate_ys.add(t)
            candidate_ys.add(t+1)
        candidate_ys = [y for y in candidate_ys if 1 <= y <= max_d]
        
        E = sorted(D)
        total = len(E)
        
        for y in candidate_ys:
            idx = bisect.bisect_left(E, y)
            F_y = total - idx
            if F_y < 1:
                continue
            snowflake_size = 1 + F_y * (y + 1)
            if snowflake_size > n:
                continue
            deletion = n - snowflake_size
            if deletion < ans:
                ans = deletion
                
    print(ans)

if __name__ == '__main__':
    main()