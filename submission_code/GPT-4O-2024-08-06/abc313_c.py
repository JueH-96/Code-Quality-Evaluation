# YOUR CODE HERE
def min_operations_to_balance(N, A):
    total_sum = sum(A)
    avg_floor = total_sum // N
    avg_ceil = (total_sum + N - 1) // N  # This is equivalent to math.ceil(total_sum / N)

    # Calculate operations to make all elements avg_floor
    operations_floor = 0
    for a in A:
        if a > avg_floor:
            operations_floor += a - avg_floor
        else:
            operations_floor += avg_floor - a

    # Calculate operations to make all elements avg_ceil
    operations_ceil = 0
    for a in A:
        if a > avg_ceil:
            operations_ceil += a - avg_ceil
        else:
            operations_ceil += avg_ceil - a

    # Since we are redistributing, we need to divide by 2 to get the actual number of operations
    operations_floor //= 2
    operations_ceil //= 2

    return min(operations_floor, operations_ceil)

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))
print(min_operations_to_balance(N, A))