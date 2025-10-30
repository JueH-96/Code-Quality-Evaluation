# YOUR CODE HERE
n, m = map(int, input().split())
a = list(map(int, input().split()))
a.sort()
total_cost = sum(a)
if total_cost <= m:
    print('infinite')
else:
    left = 0
    right = 10**9
    ans = 0
    while left <= right:
        mid = (left + right) // 2
        cost = 0
        for x in a:
            cost += min(mid, x)
        if cost <= m:
            ans = mid
            left = mid + 1
        else:
            right = mid - 1
    print(ans)