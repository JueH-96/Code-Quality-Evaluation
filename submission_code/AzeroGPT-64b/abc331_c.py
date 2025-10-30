from bisect import bisect_left, bisect_right

N = int(input())
A = list(map(int, input().split()))
A_limit = 1000001

data = [0] * (A_limit+1)
cnt = [0] * (A_limit+1)

# create BIT
def BIT_add(i, x):
    while i <= A_limit:
        data[i] += x
        cnt[i] += 1
        i += i & -i

# BIT[i-1] this is sum from 1 to i-1
def BIT_getSum(i):
    s = 0
    while i > 0:
        s += data[i]
        i -= i & -i
    return s

def BIT_getCnt(i):
    s = 0
    while i > 0:
        s += cnt[i]
        i -= i & -i
    return s

def BIT_search(x):
    # Searching is done by squaring and narrowing.
    l = 0
    r = N
    while r-l>1:
        _m = (l+r)//2
        if sum_cnt>=BIT_getCnt(_m):
            l = _m
        else:
            r = _m
    return _m

# Filling additional trees after input
for x in A:
    BIT_add(x, x)

ans = []

# solve by dividing each element
for i in range(N):
    x = A[i]
    BIT_add(x, -x)
    BIT_add(x+1, 1)  # adding 1 to avoid duplicates
    sum_sum = BIT_getSum(A_limit+1)
    sum_cnt = sum(A) - x
    ans.append(sum_sum)

print(*ans)