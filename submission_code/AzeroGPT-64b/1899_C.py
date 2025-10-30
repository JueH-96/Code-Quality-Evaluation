from typing import List

def max_subarray_with_alternating_parity(arr: List[int]) -> int:
    # This function finds the maximum sum of a subarray with alternating parities.
    sum_with_odd = [0] * len(arr)
    sum_with_even = [0] * len(arr)
    sum_with_odd[0] = arr[0] if arr[0] % 2 != 0 else 0  # Starting with an odd number or 0 if even.
    sum_with_even[0] = arr[0] if arr[0] % 2 == 0 else 0  # Starting with an even number or 0 if odd.
    max_sum = sum_with_odd[0]

    for i in range(1, len(arr)):
        if arr[i] % 2 == 0:
            sum_with_even[i] = max(sum_with_odd[i-1] + arr[i], arr[i])
        else:
            sum_with_odd[i] = max(sum_with_even[i-1] + arr[i], arr[i])
        max_sum = max(max_sum, sum_with_odd[i], sum_with_even[i])

    return max_sum

def solve():
    t = int(input())
    for _ in range(t):
        n = int(input())
        arr = list(map(int, input().split()))

        answer = max_subarray_with_alternating_parity(arr)
        print(answer)

# Solve the problem
solve()