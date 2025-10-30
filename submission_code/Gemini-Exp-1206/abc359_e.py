def solve():
    n = int(input())
    h = list(map(int, input().split()))
    
    ans = []
    for i in range(1, n + 1):
        a = [0] * (n + 1)
        count = 0
        while a[i] == 0:
            count += 1
            a[0] += 1
            for j in range(1, n + 1):
                if a[j-1] > a[j] and a[j-1] > h[j-1]:
                    a[j-1] -= 1
                    a[j] += 1
        ans.append(count)
    
    print(*ans)

solve()