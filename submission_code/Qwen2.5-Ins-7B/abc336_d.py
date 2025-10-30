# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

def check(k):
    left = [0] * (k + 1)
    right = [0] * (k + 1)
    for i in range(k):
        left[i + 1] = left[i] + (a[i] >= k - i)
        right[i + 1] = right[i] + (a[-1 - i] >= k - i)
    for i in range(k + 1):
        if left[i] + right[k - i] >= 2 * k - 1:
            return True
    return False

l, r = 1, n
while l < r:
    m = (l + r + 1) // 2
    if check(m):
        l = m
    else:
        r = m - 1
print(l)