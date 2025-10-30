def solve():
    n, s, m, l = map(int, input().split())
    
    ans = float('inf')
    
    for i in range(n // 6 + 2):
        for j in range(n // 8 + 2):
            for k in range(n // 12 + 2):
                if 6 * i + 8 * j + 12 * k >= n:
                    ans = min(ans, s * i + m * j + l * k)
    
    print(ans)

solve()