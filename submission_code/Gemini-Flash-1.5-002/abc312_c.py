# YOUR CODE HERE
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

A.sort()
B.sort()

def check(x):
    cnt_sellers = 0
    for price in A:
        if price <= x:
            cnt_sellers += 1
    cnt_buyers = 0
    for price in B:
        if price >= x:
            cnt_buyers += 1
    return cnt_sellers >= cnt_buyers

left = 1
right = 10**9 + 1
while right - left > 1:
    mid = (left + right) // 2
    if check(mid):
        right = mid
    else:
        left = mid
print(right)