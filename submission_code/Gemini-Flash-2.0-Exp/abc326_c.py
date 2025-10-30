def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    a.sort()
    
    max_gifts = 0
    for i in range(n):
        x = a[i]
        count = 0
        for j in range(n):
            if x <= a[j] < x + m:
                count += 1
        max_gifts = max(max_gifts, count)
        
    print(max_gifts)

solve()