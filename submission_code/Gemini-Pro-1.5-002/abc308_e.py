# YOUR CODE HERE
def solve():
    n = int(input())
    a = list(map(int, input().split()))
    s = input()
    
    ans = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if s[i] == 'M' and s[j] == 'E' and s[k] == 'X':
                    mex = 0
                    if mex == a[i] or mex == a[j] or mex == a[k]:
                        mex = 1
                        if mex == a[i] or mex == a[j] or mex == a[k]:
                            mex = 2
                            if mex == a[i] or mex == a[j] or mex == a[k]:
                                mex = 3
                    ans += mex
                    
    print(ans)

solve()