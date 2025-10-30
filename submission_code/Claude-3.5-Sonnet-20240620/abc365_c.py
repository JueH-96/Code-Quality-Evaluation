# YOUR CODE HERE
def solve():
    N, M = map(int, input().split())
    A = list(map(int, input().split()))

    if sum(A) <= M:
        print("infinite")
        return

    left, right = 0, 10**9 + 1
    while right - left > 1:
        mid = (left + right) // 2
        total = sum(min(a, mid) for a in A)
        if total <= M:
            left = mid
        else:
            right = mid

    print(left)

solve()