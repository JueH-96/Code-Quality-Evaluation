def solve():
    n = int(input())
    q = list(map(int, input().split()))
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    ans = 0
    for x in range(1001):
        rem_q = q[:]
        possible = True
        for i in range(n):
            rem_q[i] -= a[i] * x
            if rem_q[i] < 0:
                possible = False
                break
        
        if not possible:
            continue
        
        y = float('inf')
        for i in range(n):
            if b[i] > 0:
                y = min(y, rem_q[i] // b[i])
            elif b[i] == 0 and rem_q[i] < 0:
                y = -1
                break

        if y >= 0:
            ans = max(ans, x + int(y))
    
    print(ans)

solve()