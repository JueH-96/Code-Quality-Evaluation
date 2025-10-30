def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if a[i] * (2**a[j]) == a[j] * (2**a[i]):
                count += 1
    print(count)

t = int(input())
for _ in range(t):
    solve()