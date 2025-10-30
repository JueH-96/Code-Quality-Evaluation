import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
S = data[1]
C = list(map(int, data[2:]))

# Initialize variables to store the minimum cost
min_cost = float('inf')

# Check all possible positions to flip
for i in range(N - 1):
    if S[i] == S[i + 1]:
        # If the current pair is already good, no need to flip
        min_cost = min(min_cost, 0)
    else:
        # Calculate the cost of flipping the current pair
        cost = C[i] + C[i + 1]
        min_cost = min(min_cost, cost)

# Print the minimum cost
print(min_cost)