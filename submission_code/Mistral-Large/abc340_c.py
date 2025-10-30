import sys

def calculate_total_cost(N):
    total_cost = 0
    while N >= 2:
        total_cost += N
        N = (N // 2) + (N // 2 + (N % 2))
    return total_cost

# Read input from stdin
N = int(sys.stdin.read().strip())

# Calculate the total cost
result = calculate_total_cost(N)

# Write the output to stdout
sys.stdout.write(str(result) + '
')