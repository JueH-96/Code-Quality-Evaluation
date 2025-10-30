def solve():
    n, x = map(int, input().split())
    a = list(map(int, input().split()))

    for i in range(n-2):
        # Create a mapping from value to indices for all k > i+1
        k_indices = {}
        for k in range(i+2, n):
            if a[k] not in k_indices:
                k_indices[a[k]] = []
            k_indices[a[k]].append(k)
        
        for j in range(i+1, n-1):
            target = x - a[i] - a[j]
            if target in k_indices:
                # Find the first k > j
                for k in k_indices[target]:
                    if k > j:
                        print(i+1, j+1, k+1)  # 1-indexed
                        return

    print(-1)

solve()