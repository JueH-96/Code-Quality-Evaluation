# YOUR CODE HERE
N, T = map(int, input().split())
S = input().strip()
X = list(map(int, input().split()))

# Create list of (position, direction, index) tuples
ants = []
for i in range(N):
    ants.append((X[i], int(S[i]), i))

# Sort by position
ants.sort()

count = 0

# For each ant going right, count how many ants going left it will pass
for i in range(N):
    if ants[i][1] == 1:  # If ant is going right
        # Count ants to the right that are going left and will be passed
        for j in range(i + 1, N):
            if ants[j][1] == 0:  # If ant is going left
                # Check if they pass within time T+0.1
                distance = ants[j][0] - ants[i][0]
                if distance <= 2 * T:
                    count += 1

print(count)