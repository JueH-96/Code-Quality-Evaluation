def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    min_diff = float('inf')
    
    for i in range(1 << n):
        if bin(i).count('1') == k:
            b = []
            for j in range(n):
                if not (i >> j) & 1:
                    b.append(a[j])
            
            if len(b) > 0:
                max_b = max(b)
                min_b = min(b)
                diff = max_b - min_b
                min_diff = min(min_diff, diff)
    
    print(min_diff)

solve()