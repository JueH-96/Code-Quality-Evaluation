# YOUR CODE HERE
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Create a set for O(1) lookup of fireworks days
fireworks_days = set(A)

# Create array to store answers
answer = [0] * (N + 1)  # 1-indexed

# Process from right to left
for i in range(N, 0, -1):
    if i in fireworks_days:
        answer[i] = 0
    else:
        answer[i] = answer[i + 1] + 1

# Print answers
for i in range(1, N + 1):
    print(answer[i])