def solve():
    n = int(input())
    a = list(map(int, input().split()))
    s = input()
    
    ans = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if s[i] == 'M' and s[j] == 'E' and s[k] == 'X':
                    mex_val = mex(a[i], a[j], a[k])
                    ans += mex_val
    print(ans)

def mex(a, b, c):
    vals = {a, b, c}
    i = 0
    while i in vals:
        i += 1
    return i

solve()