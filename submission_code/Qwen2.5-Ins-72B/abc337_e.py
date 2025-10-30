import math

# Read the number of bottles
N = int(input())

# Calculate the minimum number of friends needed
M = math.ceil(math.log2(N))

# Print the number of friends to call
print(M)

# Generate the distribution of bottles to friends
for i in range(M):
    # Generate the list of bottles for the i-th friend
    bottles = [j + 1 for j in range(N) if (j >> i) & 1]
    # Print the number of bottles and the bottles themselves
    print(len(bottles), *bottles)

# Read the results from the judge
S = input().strip()

# Convert the result string to a list of integers
results = [int(c) for c in S]

# Calculate the spoiled bottle number
X = sum((1 << i) * results[i] for i in range(M))

# Print the spoiled bottle number
print(X)