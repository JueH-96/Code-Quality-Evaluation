import math

N = int(input())

count = 0
max_b = int(math.isqrt(N)) + 2  # Adding buffer to avoid missing due to rounding

for b in range(1, max_b + 1):
    if b % 2 == 0:
        continue
    b_squared = b * b
    if b_squared > N:
        continue
    max_two_power = N // b_squared
    if max_two_power < 2:
        continue
    max_a = math.floor(math.log2(max_two_power))
    # Ensure that 2^max_a <= max_two_power
    # Also, a must be >=1
    if max_a >= 1:
        count += max_a

print(count)