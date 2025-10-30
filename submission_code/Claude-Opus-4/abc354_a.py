# YOUR CODE HERE
H = int(input())

# We need to find the first day n such that 2^n - 1 > H
# This is equivalent to finding the first n such that 2^n > H + 1

n = 0
power_of_2 = 1  # This represents 2^n

while power_of_2 <= H + 1:
    n += 1
    power_of_2 *= 2

print(n)