import sys

# Read input
N, L, R = map(int, sys.stdin.readline().split())

# Function to ask a question
def ask_question(i, j):
    print(f"? {i} {j}", flush=True)
    return int(sys.stdin.readline().strip())

# Function to calculate the sum of elements in the range [L, R]
def calculate_sum(L, R):
    total_sum = 0
    i = 0
    while 2**i * (j + 1) <= 2**N:
        l = 2**i * j
        r = 2**i * (j + 1) - 1
        if r > R:
            r = R
        total_sum += ask_question(i, j)
        j = (j + 1) * (r - l + 1) // 2**i
    return total_sum

# Calculate the sum of elements in the range [L, R]
result = calculate_sum(L, R)

# Print the result
print(f"! {result % 100}", flush=True)