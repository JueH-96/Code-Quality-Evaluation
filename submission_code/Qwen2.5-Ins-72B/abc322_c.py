# YOUR CODE HERE
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Initialize the result list with zeros
result = [0] * N

# Create a set for faster lookup
fireworks_days = set(A)

# Iterate through each day and find the number of days until the next fireworks
for i in range(1, N + 1):
    if i in fireworks_days:
        result[i - 1] = 0
    else:
        for j in range(i, N + 1):
            if j in fireworks_days:
                result[i - 1] = j - i
                break

# Print the result
for r in result:
    print(r)