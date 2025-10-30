def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    
    parent = [i for i in range(N+1)]
    size = [1] * (N+1)
    color = [i for i in range(N+1)]
    count = [1] * (N+1)
    
    index = 2
    output = []
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        xroot = find(x)
        yroot = find(y)
        if xroot != yroot:
            if size[xroot] < size[yroot]:
                parent[xroot] = yroot
                size[yroot] += size[xroot]
            else:
                parent[yroot] = xroot
                size[xroot] += size[yroot]
    
    for _ in range(Q):
        query_type = int(data[index])
        if query_type == 1:
            x = int(data[index + 1])
            c = int(data[index + 2])
            index += 3
            root = find(x)
            if color[root] != c:
                old_color = color[root]
                count[old_color] -= size[root]
                color[root] = c
                count[c] += size[root]
                if x > 1:
                    left = x - 1
                    if color[find(left)] == c:
                        union(x, left)
                if x < N:
                    right = x + 1
                    if color[find(right)] == c:
                        union(x, right)
        elif query_type == 2:
            c = int(data[index + 1])
            index += 2
            output.append(str(count[c]))
    
    print('
'.join(output))

if __name__ == '__main__':
    main()