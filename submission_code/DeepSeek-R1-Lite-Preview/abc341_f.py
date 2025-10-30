def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u = int(data[idx]) - 1
        idx += 1
        v = int(data[idx]) - 1
        idx += 1
        adj[u].append(v)
        adj[v].append(u)
    
    W = [int(data[idx + i]) for i in range(N)]
    idx += N
    A = [int(data[idx + i]) for i in range(N)]
    
    vertices = sorted(range(N), key=lambda x: -W[x])
    
    O = [0] * N
    for x in vertices:
        if A[x] > 0:
            O[x] = A[x]
    
    for x in vertices:
        for y in adj[x]:
            if W[y] < W[x]:
                O[x] += O[y]
    
    total_operations = sum(O)
    print(total_operations)

if __name__ == '__main__':
    main()