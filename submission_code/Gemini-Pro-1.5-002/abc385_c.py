def solve():
    n = int(input())
    h = list(map(int, input().split()))

    ans = 0
    for height in sorted(list(set(h))):
        indices = [i for i, x in enumerate(h) if x == height]
        
        if len(indices) > ans:
            ans = max(ans, len(indices))

        for interval in range(1, n):
            for start in range(n):
                count = 0
                current = start
                while current < n:
                    if h[current] == height:
                        count += 1
                    current += interval
                ans = max(ans, count)
    print(ans)

solve()