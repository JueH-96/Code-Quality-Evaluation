# Read the first line and split into N and L
N_L = input().strip().split()
N = int(N_L[0])
L = int(N_L[1])

# Read the second line and split into scores
scores = list(map(int, input().strip().split()))

# Initialize counter
count = 0

# Iterate through scores and count those >= L
for score in scores:
    if score >= L:
        count += 1

# Print the result
print(count)