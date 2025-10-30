# YOUR CODE HERE
def max_pyramid_size(N, A):
    def is_pyramid(A, k):
        if k == 1:
            return 1 in A
        left, right = 0, len(A) - 1
        for i in range(1, k):
            while left < len(A) and A[left] < i:
                left += 1
            while right >= 0 and A[right] < i:
                right -= 1
            if left > right:
                return False
        while left <= right:
            if A[left] < k or A[right] < k:
                return False
            left += 1
            right -= 1
        return True

    left, right = 1, (N + 1) // 2
    while left < right:
        mid = (left + right + 1) // 2
        if is_pyramid(A, mid):
            left = mid
        else:
            right = mid - 1
    return left

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))
print(max_pyramid_size(N, A))