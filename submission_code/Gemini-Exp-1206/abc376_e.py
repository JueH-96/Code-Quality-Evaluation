def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    pairs = sorted(zip(a, b))
    ans = float('inf')

    for i in range(n - k + 1):
        max_a = 0
        sum_b = 0
        temp_b = []
        for j in range(i, n):
            max_a = max(max_a, pairs[j][0])
            temp_b.append(pairs[j][1])
        
        temp_b.sort(reverse=True)
        
        
        sum_b = sum(temp_b[:k])
        ans = min(ans, max_a * sum_b)

    print(ans)

t = int(input())
for _ in range(t):
    solve()