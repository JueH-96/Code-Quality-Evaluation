def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    
    n = int(data[0])
    m = int(data[1])
    
    parent = [i for i in range(n+1)]
    rank = [0] * (n+1)
    k = n
    
    def find(x):
        root = x
        while root != parent[root]:
            root = parent[root]
        temp = x
        while temp != root:
            nxt = parent[temp]
            parent[temp] = root
            temp = nxt
        return root
    
    index = 2
    for _ in range(m):
        u = int(data[index])
        v = int(data[index+1])
        index += 2
        ru = find(u)
        rv = find(v)
        if ru == rv:
            continue
        k -= 1
        if rank[ru] < rank[rv]:
            parent[ru] = rv
        elif rank[ru] > rank[rv]:
            parent[rv] = ru
        else:
            parent[rv] = ru
            rank[ru] += 1
    
    result = m - n + k
    print(result)

if __name__ == "__main__":
    main()