import math

# Read the inputs from stdin
N, S, M, L = map(int, input().split())

# Initialize the minimum cost to a large number
min_cost = float('inf')

# Try all possible combinations of buying packs
for i in range(math.ceil(N/6)+1):
    for j in range(math.ceil(N/8)+1):
        for k in range(math.ceil(N/12)+1):
            # Calculate the total number of eggs and the total cost
            total_eggs = i*6 + j*8 + k*12
            total_cost = i*S + j*M + k*L
            
            # If the total number of eggs is at least N, update the minimum cost
            if total_eggs >= N:
                min_cost = min(min_cost, total_cost)

# Print the answer
print(min_cost)