import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:2+N]))
B = list(map(int, data[2+N:2+N+M]))

A.sort()
B.sort()

def can_sell_at_price(X):
    count_sellers = 0
    count_buyers = 0
    for a in A:
        if a <= X:
            count_sellers += 1
    for b in B:
        if b >= X:
            count_buyers += 1
    return count_sellers >= count_buyers

left = 1
right = 10**9

while left < right:
    mid = (left + right) // 2
    if can_sell_at_price(mid):
        right = mid
    else:
        left = mid + 1

print(left)