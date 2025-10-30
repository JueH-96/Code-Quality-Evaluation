def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    multiples = []
    for x in a:
        if x % k == 0:
            multiples.append(x // k)
    
    multiples.sort()
    print(*multiples)

solve()