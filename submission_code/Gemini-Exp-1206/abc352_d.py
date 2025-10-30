def solve():
    n, k = map(int, input().split())
    p = list(map(int, input().split()))
    
    pos = [0] * (n + 1)
    for i in range(n):
        pos[p[i]] = i
    
    ans = float('inf')
    for start in range(1, n - k + 2):
        indices = []
        for val in range(start, start + k):
            indices.append(pos[val])
        indices.sort()
        ans = min(ans, indices[-1] - indices[0])
    
    print(ans)

solve()