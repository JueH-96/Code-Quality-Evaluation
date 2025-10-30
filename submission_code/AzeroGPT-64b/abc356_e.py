import math
from typing import List

def calculate_sum(A: List[int]) -> int:
    A = sorted([(val, i) for i, val in enumerate(A)], key=lambda x: (x[0], -x[1]))
    n = len(A)
    left_idx = [0] * n
    right_idx = [0] * n
    sum_pairs = 0

    def find_left_idx(val: int) -> int:
        low, high = 0, n - 1
        while low < high:
            mid = (low + high) // 2
            if A[mid][0] < val:
                low = mid + 1
            else:
                high = mid
        return low

    def find_right_idx(val: int, idx: int) -> int:
        low, high = idx, n - 1
        while low <= high:
            mid = (low + high) // 2
            if A[mid][0] <= val:
                low = mid + 1
            else:
                high = mid - 1
        return low

    for i, (val, _idx) in enumerate(A):
        left = find_left_idx((val - 1) // 2)
        if i > 0 and A[i - 1][0] == A[i][0]:
            left = find_left_idx(val // 2)
        right = find_right_idx(2 * val, i)
        multiplier = math.floor(val / (val / 2)) - 1

        sum_pairs += (left_idx[i - 1] * multiplier) * (n - 1 - right_idx[i - 1])

        left_idx[i] = left_idx[i - 1] + left - i
        right_idx[i] = right_idx[i - 1] + i - (right - 1)

    return sum_pairs

# Get inputs
N = int(input())
A = list(map(int, input().split()))

# Calculate and print result
print(calculate_sum(A))