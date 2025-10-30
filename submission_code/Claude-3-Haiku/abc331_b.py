import sys

def solve(N, S, M, L):
    # Initialize the minimum cost to a large value
    min_cost = float('inf')
    
    # Try all possible combinations of 6-egg, 8-egg, and 12-egg packs
    for i in range(N // 6 + 1):
        for j in range(N // 8 + 1):
            for k in range(N // 12 + 1):
                # Calculate the total number of eggs
                total_eggs = i * 6 + j * 8 + k * 12
                
                # If the total number of eggs is at least N, calculate the cost
                if total_eggs >= N:
                    cost = i * S + j * M + k * L
                    
                    # Update the minimum cost if necessary
                    min_cost = min(min_cost, cost)
    
    return min_cost

# Read the input from stdin
N, S, M, L = map(int, sys.stdin.readline().split())

# Solve the problem and print the answer
print(solve(N, S, M, L))