# N T
N, T = map(int, input().split())
# S
S = input()
# X_i
X = list(map(int, input().split()))

# Initialize counters
left_to_right = [0] * N
right_to_left = [0] * N
crossings = 0

# Calculate the number of ants that will pass each other from left to right
for i in range(N):
    if S[i] == '1':
        left_to_right[i] = X[i] + (T + 0.1)
    else:
        left_to_right[i] = -X[i] - (T + 0.1)

# Calculate the number of ants that will pass each other from right to left
for i in range(N-1, -1, -1):
    if S[i] == '0':
        right_to_left[i] = X[i] + (T + 0.1)
    else:
        right_to_left[i] = -X[i] - (T + 0.1)

# Sort the positions
left_to_right.sort()
right_to_left.sort()

# Calculate the number of crossings
for i in range(N):
    if S[i] == '1':
        crossings += sum(1 for j in range(N) if right_to_left[j] < left_to_right[i])

print(crossings)