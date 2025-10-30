# YOUR CODE HERE
N = int(input())
Q = list(map(int, input().split()))
A = list(map(int, input().split()))
B = list(map(int, input().split()))

def check(x, y):
    for i in range(N):
        if A[i] * x + B[i] * y > Q[i]:
            return False
    return True

ans = 0
for x in range(1001):
    l, r = 0, 1000001
    while l <= r:
        mid = (l + r) // 2
        if check(x, mid):
            l = mid + 1
        else:
            r = mid - 1
    ans = max(ans, x + r)

print(ans)