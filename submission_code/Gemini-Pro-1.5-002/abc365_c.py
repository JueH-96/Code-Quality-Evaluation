# YOUR CODE HERE
def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))

    total_cost = sum(a)
    if total_cost <= m:
        print("infinite")
        return

    a.sort()

    def check(x):
        cost = 0
        for val in a:
            cost += min(x, val)
        return cost <= m

    low = 0
    high = 10**9
    ans = 0

    while low <= high:
        mid = (low + high) // 2
        if check(mid):
            ans = mid
            low = mid + 1
        else:
            high = mid - 1
    
    print(ans)


solve()