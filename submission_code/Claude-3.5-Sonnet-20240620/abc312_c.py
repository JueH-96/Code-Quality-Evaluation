# YOUR CODE HERE
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

def count_sellers(price):
    return sum(1 for a in A if a <= price)

def count_buyers(price):
    return sum(1 for b in B if b >= price)

left, right = 1, 10**9 + 1

while left < right:
    mid = (left + right) // 2
    if count_sellers(mid) >= count_buyers(mid):
        right = mid
    else:
        left = mid + 1

print(left)