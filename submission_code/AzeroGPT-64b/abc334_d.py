import bisect

input_data = []
N, Q = map(int, input().split())
R = list(map(int, input().split()))
R.sort()

# 累積和
accumulated_R = [0]
for r in R:
    accumulated_R.append(accumulated_R[-1] + r)

# 二分探索で解を見つける
def binary_search(X):
    left = 0
    right = len(R)
    while right - left > 1:
        mid = (left + right) // 2
        if accumulated_R[mid] <= X:
            left = mid
        else:
            right = mid
    return left

queries = [int(input()) for _ in range(Q)]

for X in queries:
    print(binary_search(X))