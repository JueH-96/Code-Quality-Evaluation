# YOUR CODE HERE
def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    ans = 0
    for i in range(n):
        diff = (b[i] - a[i]) % m
        ans += diff
        a[i] = b[i]

        if i < n - 1:
            if a[i] == a[i+1]:
                if (a[i] + 1) % m != b[i+1]:
                    a[i+1] = (a[i+1] + 1) % m
                    ans += 1
                else:
                    a[i+1] = (a[i+1] -1) % m
                    ans += 1


    
    
    is_possible = True
    for i in range(n):
        if a[i] != b[i]:
            is_possible = False
            break
    
    if is_possible:
        print(ans)
    else:
        print(-1)


solve()