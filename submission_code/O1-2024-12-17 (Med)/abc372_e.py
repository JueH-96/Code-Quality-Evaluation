def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    
    # Fast parser
    idx = 0
    N = int(input_data[idx]); idx += 1
    Q = int(input_data[idx]); idx += 1
    
    # Disjoint Set Union (Union-Find) with "top10" tracking
    parent = list(range(N+1))
    size = [1]*(N+1)
    # top10[i] will hold up to 10 largest vertices in the connected component whose leader is i
    top10 = [[i] for i in range(N+1)]
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        # Union by size
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        # Attach smaller tree (rb) to bigger tree (ra)
        parent[rb] = ra
        size[ra] += size[rb]
        # Merge top10 lists
        merged = sorted(top10[ra] + top10[rb], reverse=True)[:10]
        top10[ra] = merged
        # Not strictly necessary, but we can clear rb's list to save memory
        top10[rb] = []
    
    answers = []
    for _ in range(Q):
        t = int(input_data[idx]); idx += 1
        if t == 1:
            u = int(input_data[idx]); idx += 1
            v = int(input_data[idx]); idx += 1
            union(u, v)
        else:
            v = int(input_data[idx]); idx += 1
            k = int(input_data[idx]); idx += 1
            rv = find(v)
            if len(top10[rv]) < k:
                answers.append(-1)
            else:
                # top10[rv] is in descending order
                answers.append(top10[rv][k-1])
    
    print("
".join(map(str, answers)))

# Call main() as required
if __name__ == "__main__":
    main()