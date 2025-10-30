def solve():
    n, k = map(int, input().split())
    s = list(input())
    
    count = 0
    i = 0
    while i <= n - k:
        if all(s[j] == 'O' for j in range(i, i + k)):
            count += 1
            for j in range(i, i + k):
                s[j] = 'X'
            i += k
        else:
            i += 1
    print(count)

solve()