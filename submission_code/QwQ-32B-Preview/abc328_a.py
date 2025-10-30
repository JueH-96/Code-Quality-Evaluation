# Read the first line and split into N and X
first_line = input().strip()
N_X = first_line.split()
N = int(N_X[0])
X = int(N_X[1])

# Read the second line and split into list of scores
scores_line = input().strip()
scores_list = scores_line.split()
S = [int(s) for s in scores_list]

# Initialize sum
total = 0

# Iterate through scores and sum up those <= X
for s in S:
    if s <= X:
        total += s

# Print the result
print(total)