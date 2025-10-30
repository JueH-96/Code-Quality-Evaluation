# Read the first line of input which contains N and L
N, L = map(int, input().split())

# Read the second line of input which contains the scores A_i
scores = list(map(int, input().split()))

# Initialize a counter for the number of people who passed
pass_count = 0

# Iterate through the scores and count how many are greater than or equal to L
for score in scores:
    if score >= L:
        pass_count += 1

# Output the number of people who passed
print(pass_count)