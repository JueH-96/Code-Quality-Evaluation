import sys

def find_minimum_cost(N, S, C):
    # Convert S to a list of integers for easier manipulation
    S = [int(char) for char in S]

    # Initialize the minimum cost to a large value
    min_cost = float('inf')

    # Function to calculate the cost of flipping the bits to make the string good
    def calculate_cost(flip_index):
        cost = 0
        for i in range(N):
            if i == flip_index or i == flip_index - 1:
                continue
            if S[i] == S[i + 1]:
                cost += min(C[i], C[i + 1])
        return cost

    # Try flipping each pair of adjacent characters
    for i in range(1, N):
        if S[i - 1] != S[i]:
            continue
        cost = calculate_cost(i)
        min_cost = min(min_cost, cost)

    return min_cost

# Read input
input = sys.stdin.read
data = input().split()

N = int(data[0])
S = data[1]
C = list(map(int, data[2:]))

# Calculate and print the minimum cost
result = find_minimum_cost(N, S, C)
print(result)