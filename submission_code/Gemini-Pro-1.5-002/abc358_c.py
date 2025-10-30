# YOUR CODE HERE
def solve():
    n, m = map(int, input().split())
    s = [input() for _ in range(n)]

    ans = float('inf')
    for i in range(1 << n):
        count = 0
        flavors = [0] * m
        for j in range(n):
            if (i >> j) & 1:
                count += 1
                for k in range(m):
                    if s[j][k] == 'o':
                        flavors[k] = 1
        
        all_flavors = True
        for k in range(m):
            if flavors[k] == 0:
                all_flavors = False
                break
        
        if all_flavors:
            ans = min(ans, count)

    print(ans)

solve()