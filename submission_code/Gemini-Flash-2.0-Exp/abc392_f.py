def solve():
    n = int(input())
    p = list(map(int, input().split()))
    
    a = []
    for i in range(n):
        a.insert(p[i] - 1, i + 1)
    
    print(*a)

solve()