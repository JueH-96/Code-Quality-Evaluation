def solve():
    n = int(input())
    p = list(map(int, input().split()))
    q = int(input())
    
    positions = {}
    for i in range(n):
        positions[p[i]] = i + 1
    
    for _ in range(q):
        a, b = map(int, input().split())
        if positions[a] < positions[b]:
            print(a)
        else:
            print(b)

solve()