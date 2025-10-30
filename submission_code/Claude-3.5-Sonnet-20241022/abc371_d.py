N = int(input())
X = list(map(int, input().split()))
P = list(map(int, input().split()))
Q = int(input())

def binary_search_left(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] >= target:
            right = mid
        else:
            left = mid + 1
    return left

def binary_search_right(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] > target:
            right = mid
        else:
            left = mid + 1
    return left

for _ in range(Q):
    L, R = map(int, input().split())
    
    # Find leftmost village index >= L
    left_idx = binary_search_left(X, L)
    # Find leftmost village index > R
    right_idx = binary_search_right(X, R)
    
    # Sum population between left_idx and right_idx-1
    total = sum(P[i] for i in range(left_idx, right_idx))
    print(total)