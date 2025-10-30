import sys

def floor_sqrt(x):
    if x == 0 or x == 1:
        return x
    left, right = 1, x
    while left <= right:
        mid = (left + right) // 2
        if mid * mid <= x:
            left = mid + 1
        else:
            right = mid - 1
    return right

def ceil_sqrt(a):
    fs = floor_sqrt(a)
    if fs * fs >= a:
        return fs
    else:
        return fs + 1

data = sys.stdin.read().strip()
N = int(data)
ans = 0
for L in range(61):  # 0 to 60 inclusive
    pow_l = 1 << L
    pow_l1 = 1 << (L + 1)
    low_s = (N // pow_l1) + 1
    high_s = N // pow_l
    if low_s <= high_s:
        A = low_s
        B = high_s
        low_k = ceil_sqrt(A)
        high_k = floor_sqrt(B)
        if low_k <= high_k:
            num_odd_k = ((high_k + 1) // 2) - (low_k // 2)
            ans += L * num_odd_k
print(ans)