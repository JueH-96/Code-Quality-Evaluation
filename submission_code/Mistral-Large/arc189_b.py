import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
X = list(map(int, data[1:]))

# Calculate the initial sum of coordinates
initial_sum = sum(X)

# Calculate the minimum possible sum after operations
min_sum = initial_sum

# Iterate over possible values of i
for i in range(N - 3):
    # Calculate the midpoint M
    M = (X[i] + X[i + 3]) / 2

    # Calculate the new positions of the (i+1)-th and (i+2)-th pieces
    new_X_i1 = 2 * M - X[i + 1]
    new_X_i2 = 2 * M - X[i + 2]

    # Calculate the new sum of coordinates
    new_sum = initial_sum - X[i + 1] - X[i + 2] + new_X_i1 + new_X_i2

    # Update the minimum sum if the new sum is smaller
    if new_sum < min_sum:
        min_sum = new_sum

# Print the minimum possible sum
print(min_sum)