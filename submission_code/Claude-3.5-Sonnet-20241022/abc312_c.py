N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

def check(x):
    sellers = sum(1 for a in A if a <= x)
    buyers = sum(1 for b in B if b >= x)
    return sellers >= buyers

left = 1
right = 10**9 + 1

while left < right:
    mid = (left + right) // 2
    if check(mid):
        right = mid
    else:
        left = mid + 1

print(left)