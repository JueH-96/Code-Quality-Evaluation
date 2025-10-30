# YOUR CODE HERE
def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    s = set(a)
    ans = 0
    for i in range(1, k + 1):
        if i not in s:
            ans += i
    print(ans)

solve()