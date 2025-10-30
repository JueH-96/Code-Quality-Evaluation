import sys

def main():
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    
    adj = [[] for _ in range(N + 1)]
    for _ in range(N - 1):
        u = int(input[idx])
        v = int(input[idx + 1])
        adj[u].append(v)
        adj[v].append(u)
        idx += 2
    
    degree = [0] * (N + 1)
    for u in range(1, N + 1):
        degree[u] = len(adj[u])
    
    global_max = 0
    
    for u in range(1, N + 1):
        cvs = []
        for v in adj[u]:
            cv = degree[v] - 1
            if cv >= 1:
                cvs.append(cv)
        cvs.sort(reverse=True)
        current_max = 0
        len_cvs = len(cvs)
        for x in range(1, len_cvs + 1):
            m = cvs[x-1]
            total = 1 + x + x * m
            if total > current_max:
                current_max = total
        if current_max > global_max:
            global_max = current_max
    
    print(N - global_max)

if __name__ == '__main__':
    main()