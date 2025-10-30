def solve():
    n = int(input())
    p = list(map(int, input().split()))
    
    max_other = 0
    for i in range(1, n):
        max_other = max(max_other, p[i])
    
    if p[0] > max_other:
        print(0)
    else:
        print(max_other - p[0] + 1)

solve()