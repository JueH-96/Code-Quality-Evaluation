N = int(input())
A = list(map(int, input().split()))

# Find target value all numbers should converge to
total = sum(A)
target = total // N
remainder = total % N

# Target values will be either target or target+1
# Numbers that should be target+1 is equal to remainder
target_high = target + 1
target_low = target

# Count operations needed
operations = 0
high_needed = remainder
low_needed = N - remainder

for a in A:
    if a > target_high:
        # Need to decrease this number
        operations += a - target_high
    elif a < target_low:
        # Need to increase this number
        operations += target_low - a

print(operations // 2)