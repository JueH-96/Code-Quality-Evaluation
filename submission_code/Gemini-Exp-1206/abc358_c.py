def solve():
    n, m = map(int, input().split())
    stands = [input() for _ in range(n)]

    min_stands = float('inf')
    for i in range(1, 1 << n):
        flavors = set()
        count = 0
        for j in range(n):
            if (i >> j) & 1:
                count += 1
                for k in range(m):
                    if stands[j][k] == 'o':
                        flavors.add(k)
        
        if len(flavors) == m:
            min_stands = min(min_stands, count)

    print(min_stands)

solve()