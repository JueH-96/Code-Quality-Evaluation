def solve():
    n = int(input())
    p = list(map(int, input().split()))
    a = list(map(int, input().split()))
    
    p = [x - 1 for x in p]
    
    min_a = list(a)
    
    current_a = list(a)
    
    for _ in range(n):
        next_a = [0] * n
        for i in range(n):
            next_a[i] = current_a[p[i]]
        
        if next_a < min_a:
            min_a = list(next_a)
        
        current_a = next_a
        
    print(*min_a)

solve()