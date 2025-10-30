def f(x):
    while x % 2 == 0:
        x //= 2
    return x

N = int(input())
A = list(map(int, input().split()))

# Precompute a_i and b_i for each A_i
# where A_i = a_i * 2^b_i, and a_i is odd
a = []  # odd part of each number
b = []  # power of 2 factor

for x in A:
    b_val = 0
    x_copy = x
    while x_copy % 2 == 0:
        x_copy //= 2
        b_val += 1
    a.append(x_copy)
    b.append(b_val)

total = 0
for i in range(N):
    for j in range(i, N):
        if b[i] == b[j]:
            # If both have same power of 2, we need to compute f of their sum
            total += f(a[i] + a[j])
        elif b[i] < b[j]:
            # Optimization based on binary representation properties
            total += a[i] + a[j] * (1 << (b[j] - b[i]))
        else:
            # Same as above but when b[i] > b[j]
            total += a[j] + a[i] * (1 << (b[i] - b[j]))

print(total)