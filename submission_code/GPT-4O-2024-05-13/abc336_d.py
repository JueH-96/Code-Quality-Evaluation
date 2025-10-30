# YOUR CODE HERE
def max_pyramid_sequence_size(N, A):
    def can_form_pyramid(k):
        # Check if we can form a pyramid sequence of size k
        left = 0
        right = N - 1
        for i in range(1, k + 1):
            if left <= right and A[left] >= i:
                left += 1
            elif left <= right and A[right] >= i:
                right -= 1
            else:
                return False
        for i in range(k - 1, 0, -1):
            if left <= right and A[left] >= i:
                left += 1
            elif left <= right and A[right] >= i:
                right -= 1
            else:
                return False
        return True

    low, high = 1, (N + 1) // 2
    result = 1

    while low <= high:
        mid = (low + high) // 2
        if can_form_pyramid(mid):
            result = mid
            low = mid + 1
        else:
            high = mid - 1

    return result

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))

print(max_pyramid_sequence_size(N, A))