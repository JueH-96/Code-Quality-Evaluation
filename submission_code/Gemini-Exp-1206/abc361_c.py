def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    ans = float('inf')
    
    for i in range(1 << n):
        removed = []
        remaining = []
        count = 0
        
        for j in range(n):
            if (i >> j) & 1:
                removed.append(a[j])
                count += 1
            else:
                remaining.append(a[j])
        
        if count != k:
            continue
        
        if not remaining:
            continue
            
        ans = min(ans, max(remaining) - min(remaining))
    
    print(ans)

solve()