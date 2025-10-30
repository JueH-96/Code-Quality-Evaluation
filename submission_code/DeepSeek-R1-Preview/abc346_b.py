import sys

# Read the input
W, B = map(int, sys.stdin.readline().split())

# Define the pattern and create an extended string
pattern = "wbwbwwbwbwbw"
S = pattern * 20  # 20 repetitions give 240 characters, which is more than enough

# Compute cumulative counts of 'w's and 'b's
cum_w = [0] * (len(S) + 1)
cum_b = [0] * (len(S) + 1)

for i in range(len(S)):
    cum_w[i+1] = cum_w[i] + (1 if S[i] == 'w' else 0)
    cum_b[i+1] = cum_b[i] + (1 if S[i] == 'b' else 0)

# Check all possible starting positions and lengths
found = False
for s in range(12):
    for l in range(1, 201):
        end = s + l
        if end > len(S):
            break
        current_w = cum_w[end] - cum_w[s]
        current_b = cum_b[end] - cum_b[s]
        if current_w == W and current_b == B:
            print("Yes")
            sys.exit()

print("No")