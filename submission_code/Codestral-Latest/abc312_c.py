import bisect

# Read input
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Sort the lists
A.sort()
B.sort()

# Binary search to find the minimum X
left, right = 1, 10**9
while left < right:
    mid = (left + right) // 2
    sellers = bisect.bisect_left(A, mid)
    buyers = bisect.bisect_right(B, mid)
    if sellers >= buyers:
        right = mid
    else:
        left = mid + 1

# Output the result
print(left)