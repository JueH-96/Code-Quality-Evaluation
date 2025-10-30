import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
M = int(data[index + 1])
L = int(data[index + 2])
index += 3
a = list(map(int, data[index:index + N]))
index += N
b = list(map(int, data[index:index + M]))
index += M
forbidden = set()
for _ in range(L):
    c = int(data[index]) - 1  # convert to 0-based
    d = int(data[index + 1]) - 1  # convert to 0-based
    forbidden.add((c, d))
    index += 2
# Sort side dishes by cost descending
sorted_sides = sorted([(b[j], j) for j in range(M)], reverse=True)
# Initialize max sum to a small value
max_sum = -1  # Minimum possible sum is 2, so -1 is safe
# For each main dish
for i in range(N):
    for bv, j in sorted_sides:
        if (i, j) not in forbidden:
            current_sum = a[i] + bv
            if current_sum > max_sum:
                max_sum = current_sum
            break  # No need to check further for this main dish
# Output the result
print(max_sum)