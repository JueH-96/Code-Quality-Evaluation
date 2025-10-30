# YOUR CODE HERE
def power_mod(base, exponent, modulus):
    result = 1
    base = base % modulus
    while exponent > 0:
        if exponent % 2 == 1:
            result = (result * base) % modulus
        exponent = exponent >> 1
        base = (base * base) % modulus
    return result

N, K = map(int, input().split())
X = list(map(int, input().split()))
A = list(map(int, input().split()))

# Create a mapping of indices
index_map = [0] * (N + 1)
for i in range(N):
    index_map[X[i]] = i + 1

# Find the cycle length for each index
cycle_lengths = [0] * (N + 1)
for i in range(1, N + 1):
    if cycle_lengths[i] == 0:
        cycle = [i]
        current = index_map[i]
        while current != i:
            cycle.append(current)
            current = index_map[current]
        cycle_length = len(cycle)
        for j in cycle:
            cycle_lengths[j] = cycle_length

# Apply the operation K times
result = [0] * N
for i in range(N):
    cycle_length = cycle_lengths[i + 1]
    final_position = power_mod(index_map[i + 1], K, cycle_length)
    current = i + 1
    for _ in range(final_position):
        current = index_map[current]
    result[i] = A[current - 1]

print(*result)