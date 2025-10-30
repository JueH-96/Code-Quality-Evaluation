# YOUR CODE HERE
def solve():
    n = int(input())
    s = input()
    a = [int(c) for c in s]
    
    ans = 0
    for i in range(n):
        for j in range(i, n):
            val = a[i]
            for k in range(i + 1, j + 1):
                val = 1 - (val * a[k])
            ans += val
    print(ans)

solve()