def solve():
    n = int(input())
    p = list(map(int, input().split()))
    a = list(map(int, input().split()))

    p = [x - 1 for x in p]

    best_a = a[:]
    
    for _ in range(n):
        new_a = [0] * n
        for i in range(n):
            new_a[i] = a[p[i]]
        
        if new_a < best_a:
            best_a = new_a[:]
        
        a = new_a[:]

    print(*best_a)

solve()