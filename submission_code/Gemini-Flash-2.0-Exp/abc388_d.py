def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    for i in range(n):
        adults = 0
        for j in range(n):
            if j < i+1 and a[j] > 0:
                adults += 1
        a[i] += adults
        
        for j in range(n):
            if j < i+1 and a[j] > 0:
                a[j] -= 1
    
    print(*a)

solve()