def solve():
    n = int(input())
    h = list(map(int, input().split()))
    
    a = [0] * (n + 1)
    ans = []
    
    for target_idx in range(1, n + 1):
        ops = 0
        a = [0] * (n + 1)
        while a[target_idx] == 0:
            ops += 1
            a[0] += 1
            for i in range(1, n + 1):
                if a[i-1] > a[i] and a[i-1] > h[i-1]:
                    a[i-1] -= 1
                    a[i] += 1
        ans.append(ops)
    print(*ans)

solve()