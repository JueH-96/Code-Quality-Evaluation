def solve():
    n, m, d = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    a.sort()
    b.sort()

    max_sum = -1
    
    i = 0
    j = 0
    while i < n:
        while j < m and b[j] < a[i] - d:
            j += 1
        
        if j < m and b[j] <= a[i] + d:
            
            k = j
            while k < m and b[k] <= a[i] + d:
                max_sum = max(max_sum, a[i] + b[k])
                k += 1
        i += 1
    
    
    
    i = 0
    j = 0
    while j < m:
        while i < n and a[i] < b[j] - d:
            i += 1
        
        if i < n and a[i] <= b[j] + d:
            
            k = i
            while k < n and a[k] <= b[j] + d:
                max_sum = max(max_sum, a[k] + b[j])
                k += 1
        j += 1

    print(max_sum)

solve()