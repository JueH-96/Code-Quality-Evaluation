def solve():
    n, k = map(int, input().split())
    s = list(input())
    
    count = 0
    i = 0
    while i <= n - k:
        if all(s[i+j] == 'O' for j in range(k)):
            count += 1
            for j in range(k):
                s[i+j] = 'X'
            i += k
        else:
            i += 1
    print(count)

solve()