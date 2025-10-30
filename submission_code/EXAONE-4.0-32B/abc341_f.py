import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    m = int(data[1])
    idx = 2
    adj = [[] for _ in range(n)]
    
    for _ in range(m):
        u = int(data[idx]) - 1
        v = int(data[idx+1]) - 1
        idx += 2
        adj[u].append(v)
        adj[v].append(u)
    
    W = list(map(int, data[idx:idx+n]))
    idx += n
    A = list(map(int, data[idx:idx+n]))
    
    incoming = [[] for _ in range(n)]
    
    for z in range(n):
        candidates = []
        for y in adj[z]:
            if W[y] < W[z]:
                candidates.append((W[y], y))
        candidates.sort(key=lambda x: (x[0], x[1]))
        total = 0
        for w, y in candidates:
            if total + w < W[z]:
                total += w
                incoming[y].append(z)
            else:
                break
                
    order = sorted(range(n), key=lambda i: (-W[i], i))
    c = [0] * n
    
    for x in order:
        c[x] = A[x]
        for z in incoming[x]:
            c[x] += c[z]
            
    total_operations = sum(c)
    print(total_operations)

if __name__ == "__main__":
    main()