# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))

total_sum = sum(A)
target = total_sum // N
remainder = total_sum % N

min_operations = 0

for num in A:
    if num > target + (1 if remainder > 0 else 0):
        min_operations += num - (target + (1 if remainder > 0 else 0))
    remainder = max(0, remainder - 1)

print(min_operations)