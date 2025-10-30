def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    count = 0
    while True:
        a.sort(reverse=True)
        
        positive_count = 0
        for x in a:
            if x > 0:
                positive_count += 1
        
        if positive_count <= 1:
            break
        
        a[0] -= 1
        a[1] -= 1
        count += 1
    
    print(count)

solve()