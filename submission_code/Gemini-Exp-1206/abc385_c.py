def solve():
    n = int(input())
    h = list(map(int, input().split()))

    ans = 1
    for height in set(h):
        for start in range(n):
            if h[start] != height:
                continue
            
            ans = max(ans, 1)
            for interval in range(1, n):
                count = 1
                current = start + interval
                while current < n:
                    if h[current] == height:
                        count += 1
                    current += interval
                ans = max(ans, count)

    print(ans)

solve()