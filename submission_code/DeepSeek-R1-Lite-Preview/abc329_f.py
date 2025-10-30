def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    
    C = list(map(int, data[2:2+N]))
    queries = data[2+N:]
    
    parent = [i for i in range(N+1)]
    size = [1] * (N+1)
    colors = [set() for _ in range(N+1)]
    
    for i in range(1, N+1):
        colors[i].add(C[i-1])
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    output = []
    for i in range(0, 2*Q, 2):
        a = int(queries[i])
        b = int(queries[i+1])
        root_a = find(a)
        root_b = find(b)
        
        if root_a != root_b:
            if size[root_a] < size[root_b]:
                parent[root_a] = root_b
                size[root_b] += size[root_a]
                colors[root_b].update(colors[root_a])
            else:
                parent[root_b] = root_a
                size[root_a] += size[root_b]
                colors[root_a].update(colors[root_b])
        
        root_b_final = find(b)
        output.append(str(len(colors[root_b_final])))
    
    print("
".join(output))

if __name__ == "__main__":
    main()