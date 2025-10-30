def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    a.sort()
    
    max_gifts = 0
    for i in range(n):
        count = 0
        for j in range(n):
            if a[i] <= a[j] < a[i] + m:
                count += 1
        max_gifts = max(max_gifts, count)
        
    
    left = 0
    current_count = 0
    
    
    for right in range(n):
        current_count +=1
        while a[right] - a[left] >= m:
            current_count -= 1
            left += 1
        
        max_gifts = max(max_gifts, current_count)
    
    print(max_gifts)

solve()