import collections

def main():
    import sys
    data = sys.stdin.read().split()
    n1 = int(data[0])
    n2 = int(data[1])
    m = int(data[2])
    n = n1 + n2
    graph = [[] for _ in range(n)]
    index = 3
    for _ in range(m):
        a = int(data[index])
        b = int(data[index + 1])
        index += 2
        a0 = a - 1
        b0 = b - 1
        graph[a0].append(b0)
        graph[b0].append(a0)
    
    d1 = [-1] * n
    q = collections.deque()
    d1[0] = 0
    q.append(0)
    while q:
        u = q.popleft()
        for v in graph[u]:
            if d1[v] == -1:
                d1[v] = d1[u] + 1
                q.append(v)
    
    d2 = [-1] * n
    d2[n - 1] = 0
    q.append(n - 1)
    while q:
        u = q.popleft()
        for v in graph[u]:
            if d2[v] == -1:
                d2[v] = d2[u] + 1
                q.append(v)
    
    max_d1 = max(d1[:n1])
    max_d2 = max(d2[n1:])
    
    print(max_d1 + 1 + max_d2)

if __name__ == "__main__":
    main()